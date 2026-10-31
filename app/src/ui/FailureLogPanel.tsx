import { useMemo, useState } from 'react';
import { formatPercent } from '../ml/metrics';
import {
  LOG_TAGS,
  MAX_LOG_ENTRIES,
  TAG_INFO,
  buildLogExport,
  logExportFileName,
  tagCounts,
  type LogEntry,
  type LogTag,
} from '../state/failureLog';
import { useSession } from '../state/session';
import { DISCLAIMER } from './disclaimer';
import { downloadJson } from './download';
import { EntryEditor } from './LogThis';

export function FailureLogPanel() {
  const { state, dispatch } = useSession();
  const log = state.failureLog;
  const [filter, setFilter] = useState<LogTag | 'all'>('all');
  const [editing, setEditing] = useState<string | null>(null);
  const counts = useMemo(() => tagCounts(log), [log]);
  const shown = useMemo(() => (filter === 'all' ? log : log.filter((e) => e.tag === filter)).slice().reverse(), [log, filter]);

  return (
    <>
      <h2>Failure log</h2>
      <p className="lead">
        Every time the model fails, write it down: which image, what it said, how sure it was, <strong>why you think it failed</strong> (the tag) and
        a hypothesis you could test. Add entries with the "Log this" buttons on the Evaluate and Break it tabs.
      </p>

      <section className="card" aria-labelledby="log-tags-title">
        <h3 id="log-tags-title">Counts by tag</h3>
        <div className="tag-filter" role="group" aria-label="Filter by tag" data-testid="log-filter">
          <button
            type="button"
            className={`tag-chip${filter === 'all' ? ' tag-chip--on' : ''}`}
            aria-pressed={filter === 'all'}
            onClick={() => setFilter('all')}
            data-testid="log-filter-all"
          >
            all <span data-testid="log-count-all">{log.length}</span>
          </button>
          {LOG_TAGS.map((t) => (
            <button
              key={t}
              type="button"
              className={`tag-chip tag-chip--${t === 'closed-world' ? 'closed' : t}${filter === t ? ' tag-chip--on' : ''}`}
              aria-pressed={filter === t}
              onClick={() => setFilter(filter === t ? 'all' : t)}
              title={TAG_INFO[t]}
              data-testid={`log-filter-${t}`}
            >
              {t} <span data-testid={`log-count-${t}`}>{counts[t]}</span>
            </button>
          ))}
        </div>
        <details className="advanced">
          <summary>What do the tags mean?</summary>
          <dl className="tag-help advanced__body" data-testid="log-tag-help">
            {LOG_TAGS.map((t) => (
              <div key={t}>
                <dt>{t}</dt>
                <dd>{TAG_INFO[t]}</dd>
              </div>
            ))}
          </dl>
        </details>
        <div className="actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => downloadJson(logExportFileName(), buildLogExport(log))}
            data-testid="log-export"
          >
            Export JSON
          </button>
          <span className="muted" data-testid="log-total">
            {log.length} of {MAX_LOG_ENTRIES} entries, saved in this browser. {DISCLAIMER}
          </span>
        </div>
      </section>

      {log.length === 0 ? (
        <div className="empty" data-testid="log-empty">
          <p>
            <strong>Nothing logged yet.</strong> Break the model first (Evaluate or Break it tab), then press "Log this" next to a failure.
          </p>
          <button type="button" className="btn btn--primary" onClick={() => dispatch({ type: 'setTab', tab: 'break' })}>
            Go to Break it
          </button>
        </div>
      ) : shown.length === 0 ? (
        <p className="muted" data-testid="log-filter-empty">
          No entries with the tag "{filter}".
        </p>
      ) : (
        <ul className="log-list" data-testid="log-list">
          {shown.map((e) => (
            <LogCard
              key={e.id}
              entry={e}
              editing={editing === e.id}
              onEdit={() => setEditing(e.id)}
              onDoneEditing={() => setEditing(null)}
              onDelete={() => {
                if (editing === e.id) setEditing(null);
                dispatch({ type: 'logDelete', id: e.id });
              }}
            />
          ))}
        </ul>
      )}
    </>
  );
}

function LogCard({
  entry: e,
  editing,
  onEdit,
  onDoneEditing,
  onDelete,
}: {
  entry: LogEntry;
  editing: boolean;
  onEdit: () => void;
  onDoneEditing: () => void;
  onDelete: () => void;
}) {
  const wrong = e.trueLabel !== null && !e.referred && e.trueLabel !== e.predictedLabel;
  return (
    <li className="log-card" data-testid="log-entry" data-tag={e.tag} data-set={e.setName}>
      <img src={e.thumbnail} alt={`Logged image from ${e.setName}`} width={96} height={96} className="log-card__img" />
      <div className="log-card__body">
        <p className="log-card__meta">
          <span className={`tag-chip tag-chip--${e.tag === 'closed-world' ? 'closed' : e.tag} tag-chip--static`} data-testid="log-entry-tag">
            {e.tag}
          </span>{' '}
          <strong>{e.setName}</strong>
          {e.imageId ? <span className="mono"> {e.imageId}</span> : null}
          <span className="muted"> - {new Date(e.createdAt).toLocaleString()}</span>
        </p>
        <p data-testid="log-entry-prediction">
          Truth: <strong>{e.trueLabel ?? 'unknown'}</strong>. Model said: <strong>{e.referred ? 'Refer to a doctor' : e.predictedLabel}</strong>
          {e.referred ? ` (leaning ${e.predictedLabel})` : ''}, {formatPercent(e.confidence, 0)} sure (P(pneumonia) = {formatPercent(e.probPneumonia)}).
          {wrong ? <strong className="wrong"> Wrong.</strong> : null}
        </p>
        {editing ? (
          <EntryEditor entry={e} onDone={onDoneEditing} />
        ) : (
          <>
            <p data-testid="log-entry-hypothesis" className={e.hypothesis ? '' : 'muted'}>
              {e.hypothesis ? <>Hypothesis: {e.hypothesis}</> : 'No hypothesis yet.'}
            </p>
            <p className="muted log-card__config">
              Settings: {e.config.mode}, {e.config.epochs} epochs, lr {e.config.lr.toExponential(1)}, seed {e.config.seed}
              {e.config.adultMix ? ', adult mix' : ''}
              {e.config.augmentation ? ', augmentation' : ''}, refer below {Math.round(e.config.tau * 100)}%,{' '}
              {e.config.usePrecomputed ? 'precomputed' : 'live'} features.
            </p>
            <div className="actions">
              <button type="button" className="btn btn--small" onClick={onEdit} data-testid="log-entry-edit">
                Edit tag and hypothesis
              </button>
              <button type="button" className="btn btn--small btn--danger" onClick={onDelete} data-testid="log-entry-delete">
                Delete
              </button>
            </div>
          </>
        )}
      </div>
    </li>
  );
}
