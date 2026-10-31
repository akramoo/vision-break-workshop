import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useReducer,
  useRef,
  type Dispatch,
  type ReactNode,
} from 'react';
import * as tf from '@tensorflow/tfjs';
import { clearFeatureCache, type FeatureProgress } from '../data/featureStore';
import { loadManifest, type ManifestEntry } from '../data/manifest';
import type { EvalSetId, SetChoice } from '../data/evalSets';
import { clearUploadStore, type UploadLabel, type UploadMeta } from '../data/uploads';
import { DEFAULT_TRAIN_SETTINGS, type TrainSettings } from '../data/trainingPlan';
import { initBackend, onWebglContextLost, type BackendInfo } from '../ml/backend';
import { loadExtractor, loadModelHash, ModelLoadError, type Extractor, type ModelHash } from '../ml/extractor';
import { bumpGeneration } from '../ml/generation';
import { createHead, type EpochStats, type Head } from '../ml/head';
import { clearSavedHead, loadSavedHead, type HeadInfo } from '../ml/headStore';
import { clampTau, MIN_TAU, type AbstainMetrics } from '../ml/metrics';
import { clearPerturbCache } from '../ml/perturbRun';
import { clearStampCache, type StampedHeadRecord } from '../ml/stampLab';
import { cancelRun } from '../ml/trainRunner';
import { liveTrackCount, stopAllStreams } from '../data/webcam';
import { MAX_LOG_ENTRIES, loadLog, saveLog, trimHypothesis, type LogEntry, type LogTag } from './failureLog';
import {
  clearPersisted,
  DEFAULT_SETTINGS,
  loadSettings,
  loadTau,
  loadTrainSettings,
  saveSettings,
  saveTau,
  saveTrainSettings,
  sanitizeTrainSettings,
  type Settings,
} from './persistence';
import { MAX_SNAPSHOTS, loadSnapshots, saveSnapshots, type Snapshot } from './snapshots';
import { loadSteps, saveSteps, toggleStepId } from './steps';

export const TABS = [
  { id: 'data', label: 'Data' },
  { id: 'train', label: 'Train' },
  { id: 'evaluate', label: 'Evaluate' },
  { id: 'break', label: 'Break it' },
  { id: 'failures', label: 'Failure log' },
  { id: 'results', label: 'Results' },
] as const;

export type TabId = (typeof TABS)[number]['id'];

export type BreakTab = 'perturb' | 'stamp' | 'images';

export interface ErrorInfo {
  title: string;
  message: string;
  hint?: string;
}

export type ModelStatus =
  | { state: 'loading'; progress: number }
  | { state: 'ready' }
  | { state: 'error'; error: ErrorInfo };

export type ManifestStatus =
  | { state: 'loading' }
  | { state: 'ready'; entries: ManifestEntry[] }
  | { state: 'error'; message: string };

/** Health of the precomputed feature files. */
export type FeatureStatus =
  | { state: 'idle' }
  | { state: 'loading'; fraction: number }
  | { state: 'ready' }
  | { state: 'failed'; reason: string };

export type TrainStatus =
  | { state: 'idle' }
  | {
      state: 'running';
      phase: 'features' | 'training';
      startedAt: number;
      epochs: number;
      featureProgress: FeatureProgress | null;
    }
  | { state: 'done' | 'cancelled'; epochs: number; completedEpochs: number; durationMs: number; keptPrevious?: boolean }
  | { state: 'error'; message: string };

export interface TrainState {
  status: TrainStatus;
  history: EpochStats[];
}

export interface RepeatRun {
  seed: number;
  /** Accuracy on our 180-image validation set. */
  val180: number | null;
  /** Accuracy on the official 16-image validation set. */
  val16: number | null;
  /** Accuracy on the Kermany test set. */
  test: number | null;
}

export interface RepeatState {
  state: 'idle' | 'running' | 'done' | 'cancelled' | 'error';
  total: number;
  /** 1-based run in progress. */
  current: number;
  epoch: number;
  runs: RepeatRun[];
  message?: string;
  featureProgress: FeatureProgress | null;
}

export interface EvalResult {
  setId: EvalSetId;
  /** The head that produced these predictions; results of other heads are stale. */
  headId: number;
  ids: string[];
  labels: number[];
  probs: Float32Array;
  /** Metrics at the time of evaluation, with `tau`. */
  metrics: AbstainMetrics;
  tau: number;
  timestamp: number;
  usedPrecomputed: boolean;
}

export interface Notice {
  id: number;
  level: 'info' | 'warn';
  message: string;
}

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
  manifest: ManifestStatus;
  features: FeatureStatus;
  trainSettings: TrainSettings;
  head: Head | null;
  headInfo: HeadInfo | null;
  train: TrainState;
  repeat: RepeatState;
  /** Selected test set in the Evaluate tab. */
  evalSelection: { set: SetChoice };
  evalResults: Partial<Record<EvalSetId, EvalResult>>;
  /** Abstain threshold; 0.5 = never refer. */
  abstainTau: number;
  /** Active sub-tab of Break it. */
  breakTab: BreakTab;
  /** Uploaded images (metadata only; pixels live in data/uploads.ts). */
  uploads: UploadMeta[];
  /** Stamp lab: the head trained on stamped images ("after"). The normal head stays in `head` ("before"). */
  stampedHead: StampedHeadRecord | null;
  /** The failure log (F8), persisted in localStorage. */
  failureLog: LogEntry[];
  /** Saved results-card snapshots (F9), persisted, at most MAX_SNAPSHOTS. */
  snapshots: Snapshot[];
  /** Ticked ids of the Workshop steps checklist, persisted. */
  stepsDone: string[];
  /** Is the Workshop steps panel expanded? (not persisted) */
  stepsOpen: boolean;
  notice: Notice | null;
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
  | { type: 'manifestReady'; entries: ManifestEntry[] }
  | { type: 'manifestFailed'; message: string }
  | { type: 'featuresStatus'; status: FeatureStatus }
  | { type: 'setTrainSettings'; patch: Partial<TrainSettings> }
  | { type: 'trainStart'; epochs: number; startedAt: number }
  | { type: 'trainFeatureProgress'; progress: FeatureProgress }
  | { type: 'trainPhase'; phase: 'features' | 'training' }
  | { type: 'trainEpoch'; stats: EpochStats }
  | { type: 'trainFinished'; state: 'done' | 'cancelled'; completedEpochs: number; durationMs: number; keptPrevious?: boolean }
  | { type: 'trainFailed'; message: string }
  | { type: 'headSet'; head: Head; info: HeadInfo; history: EpochStats[] }
  | { type: 'headCleared' }
  | { type: 'repeatStart'; total: number }
  | { type: 'repeatFeatureProgress'; progress: FeatureProgress }
  | { type: 'repeatProgress'; current: number; epoch: number }
  | { type: 'repeatRun'; run: RepeatRun }
  | { type: 'repeatEnd'; state: 'done' | 'cancelled' }
  | { type: 'repeatFailed'; message: string }
  | { type: 'setEvalSet'; set: SetChoice }
  | { type: 'setBreakTab'; tab: BreakTab }
  | { type: 'uploadsAdded'; items: UploadMeta[] }
  | { type: 'uploadLabel'; id: string; label: UploadLabel }
  | { type: 'uploadRemoved'; id: string }
  | { type: 'uploadsCleared' }
  | { type: 'stampedHeadSet'; record: StampedHeadRecord }
  | { type: 'stampedHeadCleared' }
  | { type: 'logAdd'; entry: LogEntry }
  | { type: 'logUpdate'; id: string; patch: { tag?: LogTag; hypothesis?: string } }
  | { type: 'logDelete'; id: string }
  | { type: 'snapshotAdd'; snapshot: Snapshot }
  | { type: 'snapshotDelete'; id: string }
  | { type: 'toggleStep'; id: string }
  | { type: 'setStepsOpen'; open: boolean }
  | { type: 'evalDone'; result: EvalResult }
  | { type: 'setTau'; tau: number }
  | { type: 'notice'; level: Notice['level']; message: string }
  | { type: 'dismissNotice' }
  | { type: 'resetSession' };

export const IDLE_REPEAT: RepeatState = { state: 'idle', total: 5, current: 0, epoch: 0, runs: [], featureProgress: null };

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
    manifest: { state: 'loading' },
    features: { state: 'idle' },
    trainSettings: loadTrainSettings(),
    head: null,
    headInfo: null,
    train: { status: { state: 'idle' }, history: [] },
    repeat: IDLE_REPEAT,
    evalSelection: { set: 'kermany_test' },
    evalResults: {},
    abstainTau: loadTau(),
    breakTab: 'perturb',
    uploads: [],
    stampedHead: null,
    failureLog: loadLog(),
    snapshots: loadSnapshots(),
    stepsDone: loadSteps(),
    stepsOpen: false,
    notice: null,
  };
}

let noticeCounter = 0;

export function reducer(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case 'setTab':
      return { ...state, activeTab: action.tab };
    case 'setSettingsOpen':
      return { ...state, settingsOpen: action.open };
    case 'setUsePrecomputed':
      return {
        ...state,
        settings: { ...state.settings, usePrecomputed: action.value },
        features: { state: 'idle' },
      };
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
    case 'manifestReady':
      return { ...state, manifest: { state: 'ready', entries: action.entries } };
    case 'manifestFailed':
      return { ...state, manifest: { state: 'error', message: action.message } };
    case 'featuresStatus':
      // 'ready' repeated (several loads succeed in a row) must not re-render everything.
      return action.status.state === 'ready' && state.features.state === 'ready' ? state : { ...state, features: action.status };
    case 'setTrainSettings':
      return { ...state, trainSettings: sanitizeTrainSettings({ ...state.trainSettings, ...action.patch }) };
    case 'trainStart':
      return {
        ...state,
        train: {
          status: {
            state: 'running',
            phase: 'features',
            startedAt: action.startedAt,
            epochs: action.epochs,
            featureProgress: null,
          },
          history: [],
        },
      };
    case 'trainFeatureProgress':
      return state.train.status.state === 'running'
        ? { ...state, train: { ...state.train, status: { ...state.train.status, featureProgress: action.progress } } }
        : state;
    case 'trainPhase':
      return state.train.status.state === 'running'
        ? { ...state, train: { ...state.train, status: { ...state.train.status, phase: action.phase } } }
        : state;
    case 'trainEpoch':
      return { ...state, train: { ...state.train, history: [...state.train.history, action.stats] } };
    case 'trainFinished': {
      const epochs = state.train.status.state === 'running' ? state.train.status.epochs : state.trainSettings.epochs;
      return {
        ...state,
        train: {
          ...state.train,
          status: {
            state: action.state,
            epochs,
            completedEpochs: action.completedEpochs,
            durationMs: action.durationMs,
            ...(action.keptPrevious ? { keptPrevious: true } : {}),
          },
        },
      };
    }
    case 'trainFailed':
      return { ...state, train: { ...state.train, status: { state: 'error', message: action.message } } };
    case 'headSet':
      // A new head makes every cached evaluation stale.
      return {
        ...state,
        head: action.head,
        headInfo: action.info,
        evalResults: {},
        train: action.info.restored ? { ...state.train, history: action.history } : state.train,
      };
    case 'headCleared':
      return { ...state, head: null, headInfo: null, evalResults: {} };
    case 'repeatStart':
      return { ...state, repeat: { ...IDLE_REPEAT, state: 'running', total: action.total, current: 1 } };
    case 'repeatFeatureProgress':
      return { ...state, repeat: { ...state.repeat, featureProgress: action.progress } };
    case 'repeatProgress':
      return { ...state, repeat: { ...state.repeat, current: action.current, epoch: action.epoch, featureProgress: null } };
    case 'repeatRun':
      return { ...state, repeat: { ...state.repeat, runs: [...state.repeat.runs, action.run] } };
    case 'repeatEnd':
      return { ...state, repeat: { ...state.repeat, state: action.state, featureProgress: null } };
    case 'repeatFailed':
      return { ...state, repeat: { ...state.repeat, state: 'error', message: action.message, featureProgress: null } };
    case 'setEvalSet':
      return { ...state, evalSelection: { set: action.set } };
    case 'evalDone':
      return { ...state, evalResults: { ...state.evalResults, [action.result.setId]: action.result } };
    case 'setBreakTab':
      return { ...state, breakTab: action.tab };
    case 'uploadsAdded':
      return { ...state, uploads: [...state.uploads, ...action.items] };
    case 'uploadLabel':
      return { ...state, uploads: state.uploads.map((u) => (u.id === action.id ? { ...u, label: action.label } : u)) };
    case 'uploadRemoved':
      return { ...state, uploads: state.uploads.filter((u) => u.id !== action.id) };
    case 'uploadsCleared':
      return { ...state, uploads: [] };
    case 'stampedHeadSet':
      return { ...state, stampedHead: action.record };
    case 'stampedHeadCleared':
      return { ...state, stampedHead: null };
    case 'logAdd':
      if (state.failureLog.length >= MAX_LOG_ENTRIES || state.failureLog.some((e) => e.id === action.entry.id)) return state;
      return { ...state, failureLog: [...state.failureLog, action.entry] };
    case 'logUpdate':
      return {
        ...state,
        failureLog: state.failureLog.map((e) =>
          e.id === action.id
            ? {
                ...e,
                ...(action.patch.tag !== undefined ? { tag: action.patch.tag } : {}),
                ...(action.patch.hypothesis !== undefined ? { hypothesis: trimHypothesis(action.patch.hypothesis) } : {}),
              }
            : e,
        ),
      };
    case 'logDelete':
      return { ...state, failureLog: state.failureLog.filter((e) => e.id !== action.id) };
    case 'snapshotAdd':
      return state.snapshots.length >= MAX_SNAPSHOTS ? state : { ...state, snapshots: [...state.snapshots, action.snapshot] };
    case 'snapshotDelete':
      return { ...state, snapshots: state.snapshots.filter((s) => s.id !== action.id) };
    case 'toggleStep':
      return { ...state, stepsDone: toggleStepId(state.stepsDone, action.id) };
    case 'setStepsOpen':
      return { ...state, stepsOpen: action.open };
    case 'setTau':
      return { ...state, abstainTau: clampTau(action.tau) };
    case 'notice':
      return { ...state, notice: { id: ++noticeCounter, level: action.level, message: action.message } };
    case 'dismissNotice':
      return { ...state, notice: null };
    case 'resetSession':
      // Backend and model are expensive and unaffected; everything user-made is reset.
      return {
        ...state,
        activeTab: 'data',
        settings: { ...DEFAULT_SETTINGS },
        settingsOpen: false,
        features: { state: 'idle' },
        trainSettings: { ...DEFAULT_TRAIN_SETTINGS },
        head: null,
        headInfo: null,
        train: { status: { state: 'idle' }, history: [] },
        repeat: IDLE_REPEAT,
        evalSelection: { set: 'kermany_test' },
        evalResults: {},
        abstainTau: MIN_TAU,
        breakTab: 'perturb',
        uploads: [],
        stampedHead: null,
        failureLog: [],
        snapshots: [],
        stepsDone: [],
        stepsOpen: false,
        notice: null,
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

declare global {
  interface Window {
    /** Debug hook, only installed with ?debug=1. */
    __vbw?: {
      tf: typeof tf;
      numTensors: () => number;
      getState: () => SessionState;
      /** Camera tracks started by this app that are still live (0 after Stop). */
      liveCameraTracks: () => number;
    };
  }
}

const debugMode = new URLSearchParams(globalThis.location?.search ?? '').get('debug') === '1';

export function SessionProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, undefined, initialState);
  const stateRef = useRef(state);
  stateRef.current = state;

  useEffect(() => {
    saveSettings(state.settings);
  }, [state.settings]);

  useEffect(() => {
    saveTrainSettings(state.trainSettings);
  }, [state.trainSettings]);

  // Dispose a head as soon as it is replaced or cleared.
  const prevHead = useRef<Head | null>(null);
  useEffect(() => {
    const prev = prevHead.current;
    if (prev && prev !== state.head) prev.dispose();
    prevHead.current = state.head;
  }, [state.head]);

  useEffect(() => {
    saveTau(state.abstainTau);
  }, [state.abstainTau]);

  // Failure log, snapshots and checklist survive a reload. If the browser refuses the write (storage full or blocked)
  // the data stays in memory and we say so once.
  const storageWarned = useRef({ log: false, snapshots: false });
  useEffect(() => {
    if (saveLog(state.failureLog)) {
      storageWarned.current.log = false;
    } else if (!storageWarned.current.log) {
      storageWarned.current.log = true;
      dispatch({
        type: 'notice',
        level: 'warn',
        message: 'The failure log could not be saved in this browser (storage is full or blocked). It stays in this tab until you close it: use Export JSON on the Failure log tab to keep it.',
      });
    }
  }, [state.failureLog]);
  useEffect(() => {
    if (saveSnapshots(state.snapshots)) {
      storageWarned.current.snapshots = false;
    } else if (!storageWarned.current.snapshots) {
      storageWarned.current.snapshots = true;
      dispatch({
        type: 'notice',
        level: 'warn',
        message: 'Snapshots could not be saved in this browser (storage is full or blocked). They stay in this tab until you close it: use Download results JSON on the Results tab to keep them.',
      });
    }
  }, [state.snapshots]);
  useEffect(() => {
    saveSteps(state.stepsDone);
  }, [state.stepsDone]);

  // The stamp-trained head is separate from the main head; dispose it when replaced or cleared.
  const prevStamped = useRef<StampedHeadRecord | null>(null);
  useEffect(() => {
    const prev = prevStamped.current;
    if (prev && prev !== state.stampedHead) prev.head.dispose();
    prevStamped.current = state.stampedHead;
  }, [state.stampedHead]);

  useEffect(() => {
    if (!debugMode) return;
    window.__vbw = {
      tf,
      numTensors: () => tf.memory().numTensors,
      getState: () => stateRef.current,
      liveCameraTracks: liveTrackCount,
    };
  }, []);

  useEffect(() => {
    if (bootstrapStarted) return;
    bootstrapStarted = true;
    void loadManifest().then(
      (entries) => dispatch({ type: 'manifestReady', entries }),
      (e: unknown) => dispatch({ type: 'manifestFailed', message: e instanceof Error ? e.message : String(e) }),
    );
    void loadSavedHead().then((saved) => {
      if (!saved || stateRef.current.head) return;
      dispatch({
        type: 'headSet',
        head: createHead(saved.kernel, saved.bias),
        info: { ...saved.info, restored: true },
        history: saved.history,
      });
    });
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

/**
 * Full reset: stop any run and the camera, clear the saved head (IndexedDB) and every 'vbw:' localStorage key
 * (settings, train settings, tau, log, snapshots, steps), drop the feature / sample / upload caches, and reset the
 * in-memory state. The head and the stamp-trained head are disposed by the provider when state.head / stampedHead
 * become null. The MobileNet model stays loaded.
 */
export async function resetSession(dispatch: Dispatch<Action>): Promise<void> {
  // First: jobs that are still running (or finishing) must not publish a head, save it, or dispatch after this point.
  bumpGeneration();
  cancelRun();
  stopAllStreams();
  clearFeatureCache();
  clearUploadStore();
  clearStampCache();
  clearPerturbCache();
  clearPersisted();
  dispatch({ type: 'resetSession' });
  await clearSavedHead();
}
