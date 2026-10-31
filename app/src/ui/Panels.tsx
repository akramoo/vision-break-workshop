import { TABS, useSession } from '../state/session';
import { BreakItPanel } from './BreakItPanel';
import { DataPanel } from './DataPanel';
import { EvaluatePanel } from './EvaluatePanel';
import { FailureLogPanel } from './FailureLogPanel';
import { ResultsPanel } from './ResultsPanel';
import { panelDomId, tabDomId } from './Tabs';
import { TrainPanel } from './TrainPanel';

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
      {tab.id === 'data' ? (
        <DataPanel />
      ) : tab.id === 'train' ? (
        <TrainPanel />
      ) : tab.id === 'evaluate' ? (
        <EvaluatePanel />
      ) : tab.id === 'break' ? (
        <BreakItPanel />
      ) : tab.id === 'failures' ? (
        <FailureLogPanel />
      ) : (
        <ResultsPanel />
      )}
    </section>
  );
}
