import { PAGE_SIZES, pageLabel, type PageInfo } from './paging';

interface Props {
  info: PageInfo;
  onPage: (page: number) => void;
  onPageSize: (size: number) => void;
  /** Prefix for the data-testids: `<id>-first`, `-prev`, `-next`, `-last`, `-size`, `-label`. */
  id: string;
  /** Announce page changes to screen readers (only one pager on the page should do this). Default true. */
  live?: boolean;
}

/** First / Previous / Next / Last buttons, a rows-per-page selector and a "Page x of y (rows a-b of n)" label. */
export function Pager({ info, onPage, onPageSize, id, live = true }: Props) {
  const atStart = info.page <= 1;
  const atEnd = info.page >= info.pages;
  return (
    <div className="toolbar toolbar--pager" role="group" aria-label="Pages of predictions" data-testid={`${id}-pager`}>
      <button type="button" className="btn btn--small" disabled={atStart} onClick={() => onPage(1)} data-testid={`${id}-first`} aria-label="First page">
        First
      </button>
      <button type="button" className="btn btn--small" disabled={atStart} onClick={() => onPage(info.page - 1)} data-testid={`${id}-prev`} aria-label="Previous page">
        Previous
      </button>
      <span {...(live ? { role: 'status', 'aria-live': 'polite' as const } : {})} data-testid={`${id}-label`}>
        {pageLabel(info)}
      </span>
      <button type="button" className="btn btn--small" disabled={atEnd} onClick={() => onPage(info.page + 1)} data-testid={`${id}-next`} aria-label="Next page">
        Next
      </button>
      <button type="button" className="btn btn--small" disabled={atEnd} onClick={() => onPage(info.pages)} data-testid={`${id}-last`} aria-label="Last page">
        Last
      </button>
      <label>
        Rows per page{' '}
        <select value={info.size} onChange={(e) => onPageSize(Number(e.target.value))} data-testid={`${id}-size`}>
          {PAGE_SIZES.map((n) => (
            <option key={n} value={n}>
              {n}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
