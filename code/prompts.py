"""All prompt templates in one place (shared across methods so baselines and RAG-Verify see the same wording)."""

FORMAT = (
    "Respond in exactly this format:\n"
    "Answer: <the short answer, a few words>\n"
    "Explanation: <one or two sentences justifying the answer>"
)

IDK_RULE = (
    "If the passages do not contain enough information to answer, write \"Answer: I don't know\" "
    "and briefly say what is missing."
)


def fmt_passages(passages):
    return "\n".join(f"[{i + 1}] {p['text']}" for i, p in enumerate(passages))


def closed_book(question):
    return f"Answer the following question.\n\nQuestion: {question}\n\n{FORMAT}"


def rag(question, passages, abstain=False):
    rule = f"\n{IDK_RULE}" if abstain else ""
    return (
        "Answer the question using the passages below.\n\n"
        f"Passages:\n{fmt_passages(passages)}\n\nQuestion: {question}\n\n{FORMAT}{rule}"
    )


DECOMPOSE = """Break the response below into a list of atomic claims.
Rules:
- Each claim states exactly one fact and can be understood on its own (replace pronouns with the names they refer to).
- The FIRST claim must state the answer to the question as a full sentence.
- Use only information that appears in the response; do not add or correct anything.
- Write one claim per line, starting with "- ".

Example
Question: Which river flows through the capital of Hungary?
Response: Answer: The Danube
Explanation: Budapest, the capital of Hungary, is built on both banks of the Danube.
Claims:
- The river that flows through the capital of Hungary is the Danube.
- Budapest is the capital of Hungary.
- Budapest is built on both banks of the Danube.

Example
Question: In what year was the author of "The Old Man and the Sea" awarded the Nobel Prize?
Response: Answer: 1954
Explanation: Ernest Hemingway wrote The Old Man and the Sea and received the Nobel Prize in Literature in 1954.
Claims:
- The author of "The Old Man and the Sea" was awarded the Nobel Prize in 1954.
- Ernest Hemingway wrote "The Old Man and the Sea".
- Ernest Hemingway received the Nobel Prize in Literature in 1954.

Now do the same.
Question: {question}
Response: {response}
Claims:"""


ALWAYS_ANSWER_RULE = "Always give your best answer; do not reply that you don't know."


def revise(question, passages, response, flagged, allow_idk=True):
    flagged_txt = "\n".join(f"- {c} ({status})" for c, status in flagged)
    rule = IDK_RULE if allow_idk else ALWAYS_ANSWER_RULE
    return (
        "You answered a question using the passages below. An automatic checker found that some claims in your "
        "response are not supported by the passages.\n\n"
        f"Passages:\n{fmt_passages(passages)}\n\nQuestion: {question}\n\n"
        f"Your previous response:\n{response}\n\n"
        f"Claims that failed the check:\n{flagged_txt}\n\n"
        "Rewrite the response so that every statement is supported by the passages. Correct contradicted claims, "
        f"and remove or replace unsupported ones.\n{rule}\n\n{FORMAT}"
    )


def revise_unconditional(question, passages, response):
    return (
        "You answered a question using the passages below.\n\n"
        f"Passages:\n{fmt_passages(passages)}\n\nQuestion: {question}\n\n"
        f"Your previous response:\n{response}\n\n"
        "Check your response against the passages and correct anything that is not supported by them.\n"
        f"{IDK_RULE}\n\n{FORMAT}"
    )


def revise_closed(question, response, flagged, allow_idk=True):
    flagged_txt = "\n".join(f"- {c} ({status})" for c, status in flagged)
    rule = ("If you are not confident of the answer, write \"Answer: I don't know\"." if allow_idk
            else ALWAYS_ANSWER_RULE)
    return (
        f"Question: {question}\n\nYour previous response:\n{response}\n\n"
        "A consistency check across several of your own answers flagged these claims as unreliable:\n"
        f"{flagged_txt}\n\n"
        f"Rewrite the response, correcting or removing the unreliable claims. {rule}\n\n" + FORMAT
    )


# --- Chain-of-Verification (Dhuliawala et al., 2023), factored variant
COVE_PLAN = """Below is a question and a draft response. Write up to {n} short verification questions that would check the individual facts stated in the draft. Each question must be answerable on its own, without seeing the draft. Write one question per line with no numbering.

Question: {question}
Draft response:
{draft}

Verification questions:"""

COVE_EXEC_CLOSED = "Answer the following question concisely in one sentence.\n\nQuestion: {q}"

COVE_EXEC_RAG = (
    "Answer the question concisely in one sentence using the passages below.\n\n"
    "Passages:\n{passages}\n\nQuestion: {q}"
)


def cove_final(question, draft, qa_pairs):
    qa = "\n".join(f"Q: {q}\nA: {a}" for q, a in qa_pairs)
    return (
        f"Original question: {question}\n\nDraft response:\n{draft}\n\n"
        f"Answers to independent verification questions:\n{qa}\n\n"
        "Using the verification answers, write a final response that corrects any mistakes in the draft. "
        "Where a verification answer conflicts with the draft, trust the verification answer.\n\n" + FORMAT
    )
