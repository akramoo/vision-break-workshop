import { useRef, type KeyboardEvent } from 'react';
import { TABS, useSession, type TabId } from '../state/session';

export function tabDomId(id: TabId): string {
  return `tab-${id}`;
}
export function panelDomId(id: TabId): string {
  return `panel-${id}`;
}

export function TabBar() {
  const { state, dispatch } = useSession();
  const refs = useRef<Partial<Record<TabId, HTMLButtonElement | null>>>({});

  function onKeyDown(e: KeyboardEvent, index: number) {
    const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
    if (step === 0 && e.key !== 'Home' && e.key !== 'End') return;
    e.preventDefault();
    const next = e.key === 'Home' ? 0 : e.key === 'End' ? TABS.length - 1 : (index + step + TABS.length) % TABS.length;
    const target = TABS[next];
    if (!target) return;
    dispatch({ type: 'setTab', tab: target.id });
    refs.current[target.id]?.focus();
  }

  return (
    <nav className="tabbar" aria-label="Workshop steps">
      <div role="tablist" className="tabbar__list">
        {TABS.map((tab, i) => {
          const selected = state.activeTab === tab.id;
          return (
            <button
              key={tab.id}
              ref={(el) => {
                refs.current[tab.id] = el;
              }}
              id={tabDomId(tab.id)}
              type="button"
              role="tab"
              className={`tab${selected ? ' tab--active' : ''}`}
              aria-selected={selected}
              aria-controls={panelDomId(tab.id)}
              tabIndex={selected ? 0 : -1}
              onClick={() => dispatch({ type: 'setTab', tab: tab.id })}
              onKeyDown={(e) => onKeyDown(e, i)}
            >
              {tab.label}
            </button>
          );
        })}
      </div>
    </nav>
  );
}
