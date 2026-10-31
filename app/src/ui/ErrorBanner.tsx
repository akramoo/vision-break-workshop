interface ErrorBannerProps {
  title: string;
  message: string;
  hint?: string | undefined;
}

export function ErrorBanner({ title, message, hint }: ErrorBannerProps) {
  return (
    <div className="error-banner" role="alert" data-testid="error-banner">
      <strong className="error-banner__title">{title}</strong>
      <p className="error-banner__message">{message}</p>
      {hint ? <p className="error-banner__hint">{hint}</p> : null}
    </div>
  );
}
