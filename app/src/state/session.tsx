import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  type Dispatch,
  type ReactNode,
} from 'react';
import { initBackend, onWebglContextLost, type BackendInfo } from '../ml/backend';
import { loadExtractor, loadModelHash, ModelLoadError, type Extractor, type ModelHash } from '../ml/extractor';
import { clearPersisted, DEFAULT_SETTINGS, loadSettings, saveSettings, type Settings } from './persistence';

export const TABS = [
  { id: 'data', label: 'Data', coming: 'F1' },
  { id: 'train', label: 'Train', coming: 'F2' },
  { id: 'evaluate', label: 'Evaluate', coming: 'F3' },
  { id: 'break', label: 'Break it', coming: 'F4 to F6' },
  { id: 'failures', label: 'Failure log', coming: 'F8' },
  { id: 'results', label: 'Results', coming: 'F9' },
] as const;

export type TabId = (typeof TABS)[number]['id'];

export interface ErrorInfo {
  title: string;
  message: string;
  hint?: string;
}

export type ModelStatus =
  | { state: 'loading'; progress: number }
  | { state: 'ready' }
  | { state: 'error'; error: ErrorInfo };

export interface SessionState {
  activeTab: TabId;
  settings: Settings;
  settingsOpen: boolean;
  /** null until backend detection finishes. */
  backend: BackendInfo | null;
  model: ModelStatus;
  modelHash: ModelHash | null;
  extractor: Extractor | null;
  webglLost: boolean;
}

export type Action =
  | { type: 'setTab'; tab: TabId }
  | { type: 'setSettingsOpen'; open: boolean }
  | { type: 'setUsePrecomputed'; value: boolean }
  | { type: 'backendReady'; info: BackendInfo }
  | { type: 'modelProgress'; progress: number }
  | { type: 'modelReady'; extractor: Extractor }
  | { type: 'modelFailed'; error: ErrorInfo }
  | { type: 'modelHash'; hash: ModelHash | null }
  | { type: 'webglLost' }
  | { type: 'resetSession' };

export function initialState(): SessionState {
  return {
    activeTab: 'data',
    settings: loadSettings(),
    settingsOpen: false,
    backend: null,
    model: { state: 'loading', progress: 0 },
    modelHash: null,
    extractor: null,
    webglLost: false,
  };
}

export function reducer(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case 'setTab':
      return { ...state, activeTab: action.tab };
    case 'setSettingsOpen':
      return { ...state, settingsOpen: action.open };
    case 'setUsePrecomputed':
      return { ...state, settings: { ...state.settings, usePrecomputed: action.value } };
    case 'backendReady':
      return { ...state, backend: action.info };
    case 'modelProgress':
      return state.model.state === 'loading'
        ? { ...state, model: { state: 'loading', progress: action.progress } }
        : state;
    case 'modelReady':
      return { ...state, model: { state: 'ready' }, extractor: action.extractor };
    case 'modelFailed':
      return { ...state, model: { state: 'error', error: action.error } };
    case 'modelHash':
      return { ...state, modelHash: action.hash };
    case 'webglLost':
      return { ...state, webglLost: true };
    case 'resetSession':
      // Backend and model are expensive and unaffected; everything user-made is reset.
      return {
        ...state,
        activeTab: 'data',
        settings: { ...DEFAULT_SETTINGS },
        settingsOpen: false,
      };
  }
}

interface SessionContextValue {
  state: SessionState;
  dispatch: Dispatch<Action>;
}

const SessionContext = createContext<SessionContextValue | null>(null);

// Bootstrap must run once even under React StrictMode's double effect invocation.
let bootstrapStarted = false;

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);

  useEffect(() => {
    saveSettings(state.settings);
  }, [state.settings]);

  useEffect(() => {
    if (bootstrapStarted) return;
    bootstrapStarted = true;
    void (async () => {
      const info = await initBackend();
      dispatch({ type: 'backendReady', info });
      if (info.backend === 'none') {
        dispatch({
          type: 'modelFailed',
          error: {
            title: 'No compute backend available',
            message: info.reason ?? 'TensorFlow.js could not start in this browser.',
            hint: 'Try the latest Chrome, Edge or Firefox.',
          },
        });
        return;
      }
      onWebglContextLost(() => dispatch({ type: 'webglLost' }));
      void loadModelHash().then((hash) => dispatch({ type: 'modelHash', hash }));
      try {
        const extractor = await loadExtractor((progress) => dispatch({ type: 'modelProgress', progress }));
        dispatch({ type: 'modelReady', extractor });
      } catch (e) {
        const error: ErrorInfo =
          e instanceof ModelLoadError
            ? { title: 'Model not available', message: e.message, hint: e.hint }
            : { title: 'Model failed to load', message: e instanceof Error ? e.message : String(e) };
        dispatch({ type: 'modelFailed', error });
      }
    })();
  }, []);

  const value = useMemo(() => ({ state, dispatch }), [state]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): SessionContextValue {
  const ctx = useContext(SessionContext);
  if (!ctx) throw new Error('useSession must be used inside <SessionProvider>');
  return ctx;
}

/** Clear persisted data and reset in-memory session state. */
export function resetSession(dispatch: Dispatch<Action>): void {
  clearPersisted();
  dispatch({ type: 'resetSession' });
}
