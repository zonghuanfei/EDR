"""Retry an incomplete solver or query call once with the same request."""
import gzip
import json
import re
import time


def load(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


class TechnicalRetry:
    def __init__(self, library):
        self.library = library
        self.original = library.api_call

    def invalid_reason(self, path, payload, raw):
        if '_probe_' in path.name or '_retrieval_' in path.name:
            return None
        is_solver = bool(re.fullmatch(r'round_\d+_solver(?:_independent|_discussion)?\.json\.gz', path.name))
        is_query = bool(re.fullmatch(r'round_\d+_query\.json\.gz', path.name))
        if not is_solver and not is_query:
            return None
        try:
            choices = raw.get('choices', [])
            if len(choices) != payload['n']:
                return 'wrong_candidate_count'
            if is_query:
                self.library.parse_queries(choices[0])
            else:
                for choice in choices:
                    text = self.library.complete_choice(choice)
                    if self.library.parse_answer(text, set('ABCDEFGHIJ')) is None:
                        return 'no_valid_final_answer_tag'
        except (RuntimeError, ValueError, KeyError, TypeError) as error:
            return f'{type(error).__name__}: {error}'
        return None

    def __call__(self, path, url, payload, source_path=None):
        # Frozen source responses, including failures, are inherited unchanged.
        # Regenerating one would change the paired initial materials.
        if source_path is not None:
            return self.original(path, url, payload, source_path=source_path)
        failed_path = path.with_name(path.name.removesuffix('.json.gz') + '.failed_attempt0.json.gz')
        retry_path = path.with_name(path.name.removesuffix('.json.gz') + '.retry1.json.gz')
        intent_path = path.with_name(path.name + '.retry_intent.json')
        if path.exists():
            record = load(path)
            if record.get('technical_retry_resolution'):
                return self.original(path, url, payload)
        elif failed_path.exists():
            if not retry_path.exists():
                raise RuntimeError('Prior uncertain retry requires inspection; do not silently issue another attempt')
            retried = self.original(retry_path, url, payload)
            self.library.save_receipt(path, dict(request=payload, response=retried, derived=True,
                                                technical_retry_resolution=True, resolved_from=str(retry_path),
                                                failed_source=str(failed_path), same_request=True))
            return retried
        raw = self.original(path, url, payload, source_path=source_path)
        reason = self.invalid_reason(path, payload, raw)
        if reason is None:
            return raw
        record = load(path)
        if record.get('derived'):
            raise RuntimeError('Invalid merged output must be rebuilt from its original channel calls')
        if failed_path.exists():
            raise RuntimeError('Refusing to overwrite a prior failed-attempt archive')
        path.rename(failed_path)
        self.library.write_json(intent_path,
                                dict(time=time.time(), reason=reason, request=payload,
                                     policy='one_same_parameter_retry_per_incomplete_call',
                                     failed_source=str(failed_path), maximum_additional_attempts=1))
        # The same n, prompt, sampling and token limit are preserved; do not keep
        # seven choices and resample only the eighth one from a failed batch.
        retried = self.original(retry_path, url, payload)
        resolved = dict(request=payload, response=retried, derived=True, technical_retry_resolution=True,
                        resolved_from=str(retry_path), failed_source=str(failed_path),
                        first_failure=reason, same_request=True)
        self.library.save_receipt(path, resolved)
        return retried
