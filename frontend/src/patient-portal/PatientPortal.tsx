import React from "react";
import { IntlProvider, FormattedMessage } from "react-intl";

type Props = {
  locale: "en" | "es";
  messages: Record<string, string>;
  countdownDays: number;
  onReorder: () => void;
};

export function PatientPortal({ locale, messages, countdownDays, onReorder }: Props) {
  return (
    <IntlProvider locale={locale} messages={messages}>
      <main style={{ maxWidth: 520, margin: "0 auto", padding: 16, fontSize: 18, lineHeight: 1.5, color: "#111", backgroundColor: "#fff" }}>
        <h1 style={{ fontSize: 32 }}>
          <FormattedMessage id="portal.title" defaultMessage="Your Supply Portal" />
        </h1>
        <p>
          <FormattedMessage id="portal.countdown" defaultMessage="Your catheter supply will last ~{days} more days" values={{ days: countdownDays }} />
        </p>
        <button onClick={onReorder} style={{ width: "100%", fontSize: 22, padding: 14, backgroundColor: "#004aad", color: "#fff", borderRadius: 8, border: "none" }}>
          <FormattedMessage id="portal.reorder" defaultMessage="Reorder Now" />
        </button>
      </main>
    </IntlProvider>
  );
}
