import { DISCLAIMER } from './disclaimer';

export function Footer() {
  return (
    <footer className="app-footer">
      <p className="disclaimer" data-testid="disclaimer-footer">
        {DISCLAIMER}
      </p>
      <p className="app-footer__note">Runs entirely in your browser. X-rays and models never leave this computer.</p>
    </footer>
  );
}
