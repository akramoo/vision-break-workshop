import { TABS, useSession } from '../state/session';
import { panelDomId, tabDomId } from './Tabs';

// Placeholder panels; each step (F1..F9) replaces its entry.
export function ActivePanel() {
  const { state } = useSession();
  const tab = TABS.find((t) => t.id === state.activeTab) ?? TABS[0];
  return (
    <section
      id={panelDomId(tab.id)}
      role="tabpanel"
      aria-labelledby={tabDomId(tab.id)}
      className="panel"
      data-testid={`panel-${tab.id}`}
    >
      <h2>{tab.label}</h2>
      <p className="panel__placeholder">Coming in {tab.coming}.</p>
    </section>
  );
}
