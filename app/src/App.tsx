import { MemoryHud } from './dev/MemoryHud';
import { useSession } from './state/session';
import { ErrorBanner } from './ui/ErrorBanner';
import { Footer } from './ui/Footer';
import { Header } from './ui/Header';
import { ActivePanel } from './ui/Panels';
import { SettingsDrawer } from './ui/SettingsDrawer';
import { TabBar } from './ui/Tabs';

const debug = new URLSearchParams(window.location.search).get('debug') === '1';

export function App() {
  const { state } = useSession();
  return (
    <div className="app">
      <Header />
      <TabBar />
      <main className="app-main">
        {state.model.state === 'error' ? (
          <ErrorBanner
            title={state.model.error.title}
            message={state.model.error.message}
            hint={state.model.error.hint}
          />
        ) : null}
        {state.webglLost ? (
          <ErrorBanner
            title="WebGL context lost"
            message="The graphics context was reset, so computations may fail."
            hint="Reload the page, or open it with ?backend=wasm to avoid the GPU."
          />
        ) : null}
        <ActivePanel />
      </main>
      <Footer />
      <SettingsDrawer />
      {debug ? <MemoryHud /> : null}
    </div>
  );
}
