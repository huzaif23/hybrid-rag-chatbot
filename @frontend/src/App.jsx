import { useState } from "react";

const API_CHAT = "/api/chat";
const API_FEEDBACK = "/api/feedback";

function App() {
  const [input, setInput] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [sources, setSources] = useState([]);
  const [error, setError] = useState(null);
  const [feedbackTarget, setFeedbackTarget] = useState(null);
  /** null = awaiting choice; "saved" = helpful recorded; "dismissed" = not helpful, no API */
  const [feedbackOutcome, setFeedbackOutcome] = useState(null);
  const [feedbackError, setFeedbackError] = useState(false);
  const [feedbackSubmitting, setFeedbackSubmitting] = useState(false);

  const handleAsk = async () => {
    if (!input.trim()) return;

    const submittedQuery = input.trim();
    setLoading(true);
    setError(null);
    setFeedbackTarget(null);
    setFeedbackOutcome(null);
    setFeedbackError(false);
    setResponse("Thinking...");

    try {
      const res = await fetch(API_CHAT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: submittedQuery }),
      });

      const data = await res.json();

      const answerText = data.answer || "No answer found.";

      if (data.is_tax_topic === false) {
        setResponse(
          data.answer ||
            "I only answer tax-related questions based on IRS data.",
        );
        setSources([]);
      } else {
        setResponse(answerText);
        setSources(data.sources || []);
      }

      // Feedback only for successful unsourced fallback; RAG uses IRS chunks, and
      // fallback_verification_rejected is a stub, not a rated answer.
      const src = data.answer_source || "";
      const isSuccessfulFallback = src === "fallback";

      if (isSuccessfulFallback) {
        setFeedbackTarget({
          query: submittedQuery,
          answer: answerText,
        });
      } else {
        setFeedbackTarget(null);
      }
    } catch (e) {
      setError(
        "Failed to connect to backend. Make sure the server is running on port 8000.",
      );
    } finally {
      setLoading(false);
      setInput("");
    }
  };

  const submitHelpful = async () => {
    if (!feedbackTarget || feedbackOutcome !== null || feedbackSubmitting)
      return;
    setFeedbackSubmitting(true);
    setFeedbackError(false);
    try {
      const res = await fetch(API_FEEDBACK, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: feedbackTarget.query,
          answer: feedbackTarget.answer,
          helpful: true,
        }),
      });
      if (!res.ok) {
        throw new Error("feedback failed");
      }
      const data = await res.json().catch(() => ({}));
      if (data.stored === false) {
        throw new Error("feedback not stored");
      }
      setFeedbackOutcome("saved");
    } catch (e) {
      setFeedbackError(true);
    } finally {
      setFeedbackSubmitting(false);
    }
  };

  /** Not helpful: acknowledge only — no network, no storage (per product rules). */
  const acknowledgeNotHelpful = () => {
    if (!feedbackTarget || feedbackOutcome !== null || feedbackSubmitting)
      return;
    setFeedbackOutcome("dismissed");
  };

  const showFeedback =
    feedbackTarget &&
    !loading &&
    response &&
    response !== "Thinking..." &&
    !error;

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <img
            src="/zyvzen-logo.png"
            alt="Zyvzen"
            style={styles.headerWordmark}
          />
        </div>

        <h1 style={styles.title}>Tax Assistant AI</h1>

        <div style={styles.inputContainer}>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleAsk()}
            placeholder="Ask a tax question (e.g., 'How do I file Form 1040?')"
            disabled={loading}
            style={styles.input}
          />
          <button
            onClick={handleAsk}
            disabled={loading}
            style={{
              ...styles.button,
              ...(loading ? styles.buttonDisabled : {}),
            }}
          >
            Ask
          </button>
        </div>

        {loading && <p style={styles.loading}>Thinking...</p>}

        {error && <p style={styles.error}>{error}</p>}

        {response && !loading && (
          <div style={styles.response}>
            <h2 style={styles.responseTitle}>Answer:</h2>
            <div style={styles.responseText}>{response}</div>

            {sources.length > 0 && (
              <div style={styles.sources}>
                <h3 style={styles.sourcesTitle}>Sources:</h3>
                <ul style={styles.sourcesList}>
                  {sources.map((url, i) => (
                    <li key={i} style={styles.sourceItem}>
                      <a href={url} target="_blank" rel="noopener noreferrer">
                        {url}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {showFeedback && (
              <div style={styles.feedback}>
                <p style={styles.feedbackPrompt}>Was this helpful?</p>
                {feedbackOutcome === null ? (
                  <div style={styles.feedbackRow}>
                    <button
                      type="button"
                      onClick={submitHelpful}
                      disabled={feedbackSubmitting}
                      style={styles.feedbackBtnYes}
                    >
                      [Yes] Helpful
                    </button>
                    <button
                      type="button"
                      onClick={acknowledgeNotHelpful}
                      disabled={feedbackSubmitting}
                      style={styles.feedbackBtnNo}
                    >
                      [No] Not Helpful
                    </button>
                  </div>
                ) : (
                  <p
                    style={
                      feedbackOutcome === "saved"
                        ? styles.feedbackThanks
                        : styles.feedbackDismissed
                    }
                  >
                    {feedbackOutcome === "saved"
                      ? "Thanks — your feedback was saved."
                      : "Thanks for letting us know."}
                  </p>
                )}
                {feedbackError && (
                  <p style={styles.feedbackErr}>
                    Could not save feedback. Check that the API is running.
                  </p>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      <footer style={styles.footer}>
        <p style={styles.footerLead}>
          This AI-powered Hybrid RAG chatbot showcases our approach to building
          fast, accurate, and scalable intelligent systems. Currently in MVP
          phase, it already demonstrates the core capabilities of high-precision
          retrieval and contextual responses. The production version will
          deliver significantly enhanced speed, accuracy, and reliability.
        </p>
        <p style={styles.footerContact}>
          For collaboration or inquiries:{" "}
          <a href="mailto:contact@zyvzen.com" style={styles.footerLink}>
            contact@zyvzen.com
          </a>
        </p>
      </footer>
    </div>
  );
}

const styles = {
  page: {
    maxWidth: "640px",
    margin: "40px auto",
    padding: "0 16px 48px",
    fontFamily: "system-ui, -apple-system, sans-serif",
  },
  card: {
    padding: "24px",
    backgroundColor: "#f9fafb",
    borderRadius: "12px",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
  },
  cardHeader: {
    display: "flex",
    justifyContent: "flex-start",
    marginBottom: "12px",
  },
  headerWordmark: {
    display: "block",
    height: "36px",
    width: "auto",
    maxWidth: "100%",
    objectFit: "contain",
    objectPosition: "left center",
  },
  title: {
    textAlign: "center",
    marginTop: 0,
    marginBottom: "20px",
    color: "#1f2937",
  },
  inputContainer: {
    display: "flex",
    gap: "10px",
    marginBottom: "16px",
  },
  input: {
    flex: 1,
    padding: "12px 16px",
    fontSize: "16px",
    border: "1px solid #d1d5db",
    borderRadius: "8px",
    outline: "none",
  },
  button: {
    padding: "12px 24px",
    fontSize: "16px",
    fontWeight: 600,
    backgroundColor: "#0a0a0a",
    color: "#ffffff",
    border: "1px solid #000000",
    borderRadius: "8px",
    cursor: "pointer",
    boxShadow: "0 1px 2px rgba(0,0,0,0.12)",
  },
  buttonDisabled: {
    opacity: 0.5,
    cursor: "not-allowed",
  },
  loading: {
    textAlign: "center",
    color: "#6b7280",
    marginBottom: "16px",
  },
  error: {
    textAlign: "center",
    color: "#dc2626",
    marginBottom: "16px",
    backgroundColor: "#fef2f2",
    padding: "12px",
    borderRadius: "8px",
  },
  response: {
    backgroundColor: "white",
    padding: "20px",
    borderRadius: "8px",
    border: "1px solid #e5e7eb",
  },
  responseTitle: {
    marginTop: 0,
    marginBottom: "12px",
    fontSize: "16px",
    color: "#374151",
  },
  responseText: {
    fontSize: "14px",
    lineHeight: "1.6",
    color: "#1f2937",
    whiteSpace: "pre-wrap",
  },
  sources: {
    marginTop: "16px",
    paddingTop: "16px",
    borderTop: "1px solid #e5e7eb",
  },
  sourcesTitle: {
    fontSize: "14px",
    color: "#6b7280",
    marginBottom: "8px",
  },
  sourcesList: {
    listStyle: "none",
    padding: 0,
    margin: 0,
  },
  sourceItem: {
    fontSize: "12px",
    color: "#2563eb",
    marginBottom: "4px",
  },
  feedback: {
    marginTop: "20px",
    paddingTop: "16px",
    borderTop: "1px solid #e5e7eb",
  },
  feedbackPrompt: {
    margin: "0 0 10px 0",
    fontSize: "14px",
    color: "#374151",
    fontWeight: 600,
  },
  feedbackRow: {
    display: "flex",
    gap: "10px",
    flexWrap: "wrap",
  },
  feedbackBtnYes: {
    padding: "8px 16px",
    fontSize: "14px",
    backgroundColor: "#059669",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
  },
  feedbackBtnNo: {
    padding: "8px 16px",
    fontSize: "14px",
    backgroundColor: "#6b7280",
    color: "white",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
  },
  feedbackThanks: {
    margin: 0,
    fontSize: "14px",
    color: "#059669",
  },
  feedbackDismissed: {
    margin: 0,
    fontSize: "14px",
    color: "#6b7280",
  },
  feedbackErr: {
    margin: "8px 0 0 0",
    fontSize: "13px",
    color: "#dc2626",
  },
  footer: {
    marginTop: "28px",
  },
  footerLead: {
    margin: "0 0 16px 0",
    fontSize: "14px",
    lineHeight: 1.65,
    color: "#4b5563",
  },
  footerContact: {
    margin: 0,
    fontSize: "14px",
    lineHeight: 1.5,
    color: "#374151",
    fontWeight: 500,
  },
  footerLink: {
    color: "#0a0a0a",
    textDecoration: "underline",
    textUnderlineOffset: "2px",
  },
};

export default App;
