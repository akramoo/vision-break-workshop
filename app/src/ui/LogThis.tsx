// Inline "Log this" form (F8): tag dropdown with a one-line explanation of every tag, hypothesis, Save.
import { useId, useState } from 'react';
import { modelConfigOf } from '../ml/modelConfig';
import {
  LOG_TAGS,
  MAX_HYPOTHESIS_CHARS,
  MAX_LOG_ENTRIES,
  TAG_INFO,
  buildLogConfig,
  isLogTag,
  makeLogEntry,
  type LogConfig,
  type LogDraftBase,
  type LogEntry,
  type LogTag,
} from '../state/failureLog';
import { useSession } from '../state/session';

/** Settings snapshot builder for the current session: model config + tau + feature mode (+ context extras). */
export function useLogConfig(): (extra?: Record<string, unknown>) => LogConfig {
  const { state } = useSession();
  return (extra) =>
    buildLogConfig(modelConfigOf(state.headInfo, state.trainSettings), state.abstainTau, state.settings.usePrecomputed, extra);
}

/** Add an entry to the log, or explain why not (log full). Returns the new entry. */
export function useAddLog(): (draft: LogDraftBase, tag: LogTag, hypothesis: string) => LogEntry | null {
  const { state, dispatch } = useSession();
  return (draft, tag, hypothesis) => {
    if (state.failureLog.length >= MAX_LOG_ENTRIES) {
      dispatch({
        type: 'notice',
        level: 'warn',
        message: `The failure log is full (${MAX_LOG_ENTRIES} entries). Export it and delete some entries first.`,
      });
      return null;
    }
    const entry = makeLogEntry({ ...draft, tag, hypothesis });
    dispatch({ type: 'logAdd', entry });
    return entry;
  };
}

export function TagSelect({ value, onChange, id }: { value: LogTag | ''; onChange: (t: LogTag | '') => void; id: string }) {
  return (
    <select
      id={id}
      value={value}
      data-testid="log-tag-select"
      onChange={(e) => onChange(isLogTag(e.target.value) ? e.target.value : '')}
    >
      <option value="">Choose a tag...</option>
      {LOG_TAGS.map((t) => (
        <option key={t} value={t}>
          {t}: {TAG_INFO[t]}
        </option>
      ))}
    </select>
  );
}

interface FormProps {
  /** Builds the entry's fixed part (thumbnail, prediction, config) when Save is pressed. */
  build: () => Promise<LogDraftBase>;
  onSaved: (entry: LogEntry) => void;
  onCancel: () => void;
}

/** Tag + hypothesis form for a new entry. */
export function LogForm({ build, onSaved, onCancel }: FormProps) {
  const uid = useId();
  const add = useAddLog();
  const [tag, setTag] = useState<LogTag | ''>('');
  const [hypothesis, setHypothesis] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function save() {
    if (tag === '') return;
    setBusy(true);
    setError(null);
    try {
      const entry = add(await build(), tag, hypothesis);
      if (entry) onSaved(entry);
      else onCancel();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form
      className="log-form"
      data-testid="log-form"
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      <div className="log-form__row">
        <label htmlFor={`${uid}-tag`}>Why did it fail? (tag)</label>
        <TagSelect id={`${uid}-tag`} value={tag} onChange={setTag} />
      </div>
      <p className="muted log-form__explain" data-testid="log-tag-explain">
        {tag === '' ? 'Pick the tag that fits best. Each one is explained in the dropdown.' : `${tag}: ${TAG_INFO[tag]}`}
      </p>
      <div className="log-form__row">
        <label htmlFor={`${uid}-hyp`}>Your hypothesis (why do you think it happened?)</label>
        <textarea
          id={`${uid}-hyp`}
          data-testid="log-hypothesis"
          rows={2}
          maxLength={MAX_HYPOTHESIS_CHARS}
          value={hypothesis}
          placeholder="For example: the training set has three times more pneumonia, so the model guesses pneumonia when unsure."
          onChange={(e) => setHypothesis(e.target.value)}
        />
      </div>
      {error ? (
        <p className="inline-warn inline-warn--strong" role="alert">
          {error}
        </p>
      ) : null}
      <div className="actions">
        <button type="submit" className="btn btn--primary btn--small" disabled={tag === '' || busy} data-testid="log-save">
          {busy ? 'Saving...' : 'Save to failure log'}
        </button>
        <button type="button" className="btn btn--small" onClick={onCancel} data-testid="log-cancel">
          Cancel
        </button>
      </div>
    </form>
  );
}

interface ThisProps {
  build: () => Promise<LogDraftBase>;
  testId?: string;
  label?: string;
  disabled?: boolean;
}

/** "Log this" button that opens the inline form; after saving it shows a short confirmation. */
export function LogThis({ build, testId = 'log-this', label = 'Log this', disabled = false }: ThisProps) {
  const [open, setOpen] = useState(false);
  const [saved, setSaved] = useState(0);
  return (
    <span className="log-this">
      <button
        type="button"
        className="btn btn--small"
        aria-expanded={open}
        disabled={disabled}
        data-testid={testId}
        onClick={() => setOpen((o) => !o)}
      >
        {label}
      </button>
      {saved > 0 && !open ? (
        <span className="log-this__saved" role="status" data-testid={`${testId}-saved`}>
          {' '}
          Saved ({saved})
        </span>
      ) : null}
      {open ? (
        <LogForm
          build={build}
          onCancel={() => setOpen(false)}
          onSaved={() => {
            setSaved((n) => n + 1);
            setOpen(false);
          }}
        />
      ) : null}
    </span>
  );
}

/** Edit the tag and hypothesis of an existing entry (used in the Failure log and right after a one-click log). */
export function EntryEditor({ entry, onDone, doneLabel = 'Save' }: { entry: LogEntry; onDone: () => void; doneLabel?: string }) {
  const uid = useId();
  const { dispatch } = useSession();
  const [tag, setTag] = useState<LogTag | ''>(entry.tag);
  const [hypothesis, setHypothesis] = useState(entry.hypothesis);
  return (
    <form
      className="log-form"
      data-testid="log-edit-form"
      onSubmit={(e) => {
        e.preventDefault();
        if (tag === '') return;
        dispatch({ type: 'logUpdate', id: entry.id, patch: { tag, hypothesis } });
        onDone();
      }}
    >
      <div className="log-form__row">
        <label htmlFor={`${uid}-tag`}>Tag</label>
        <TagSelect id={`${uid}-tag`} value={tag} onChange={setTag} />
      </div>
      <p className="muted log-form__explain">{tag === '' ? 'Pick a tag.' : `${tag}: ${TAG_INFO[tag]}`}</p>
      <div className="log-form__row">
        <label htmlFor={`${uid}-hyp`}>Hypothesis</label>
        <textarea
          id={`${uid}-hyp`}
          data-testid="log-edit-hypothesis"
          rows={2}
          maxLength={MAX_HYPOTHESIS_CHARS}
          value={hypothesis}
          onChange={(e) => setHypothesis(e.target.value)}
        />
      </div>
      <div className="actions">
        <button type="submit" className="btn btn--primary btn--small" disabled={tag === ''} data-testid="log-edit-save">
          {doneLabel}
        </button>
        <button type="button" className="btn btn--small" onClick={onDone} data-testid="log-edit-cancel">
          Cancel
        </button>
      </div>
    </form>
  );
}
