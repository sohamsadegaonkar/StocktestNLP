# Evaluation and jury demo

## Demonstrate in five minutes

1. Open the inbox and select **Agreement to acquire in-tech**. Show the **Agreement signed** stage, the EUR 450 million amount, and the original investor filing link. Select Tata Motors and explain that its demerger was still subject to approvals in the source used.
2. Open **Edge lab**. Disconnect the simulated source link and advance two packets. The virtual display and queue show two pending packets. Both carry the same historical RBI announcement to demonstrate retransmission.
3. Open **Analyze**, load the example or paste a disclosure, then analyze it while the link is down. It appears in the local inbox. Restarting the application with the same SQLite path preserves the pending queue and the local brief.
4. Reconnect the source link. The RBI disclosure appears once, pending falls to zero, and the duplicate count increases to one. Advance the remaining packets to see the separately marked synthetic scenarios.
5. Run **Evaluation**. Compare macro-F1 and review recall, read the confusion matrix, and explain the cost of accepting more cases. Open the historical results to verify category and stage against the source links.

## Explain the engineering

- **ML** A quantized BGE transformer turns a passage into a vector representing its meaning. A supervised classifier learns event labels from those vectors. Word and character TF-IDF and keyword matching are benchmark baselines.
- **NLP** Page-aware segmentation preserves character offsets. Document and chunk scores determine the category and strongest supporting evidence. Rules extract disclosed amounts and guard basic stage distinctions such as signed, completed, denied, and subject to approvals.
- **Edge** CPU inference, file parsing, and the SQLite inbox all run locally. The link simulation controls delivery of inbound packets only. No cloud API is used for a brief and no edge kit is needed for demonstration.
- **AI comparison** Optional INT8 MobileBERT evaluates whether a source chunk entails each plain-language event description. It supports an independent second opinion and a repeatable comparison once the model artifacts are installed.

## Interpretation

Macro-F1 gives each labeled category and the review class equal weight. Coverage is the fraction assigned a substantive category. Accuracy when covered is computed over those predictions. Review recall measures the proportion of ambiguous examples routed to review. A model can increase coverage by reviewing fewer cases and lose review recall at the same time.

The current training set, calibration set, and holdout were authored for this project. The four historical examples are curated source-backed checks. A meaningful follow-up evaluation would sample unseen NSE filings across issuers and time periods, use two independent reviewers, and resolve label disagreements before scoring.

The browser demo and one-CPU benchmark run on a workstation. This version implements a deterministic link simulator and virtual bitmap. Physical board throughput, power, thermal behavior, and display integration require separate measurement.
