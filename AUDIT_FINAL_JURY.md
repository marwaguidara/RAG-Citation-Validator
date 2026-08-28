# AUDIT FINAL — RAG Citation Validator
## Évaluation de niveau jury de PFE — Master IA / AI Engineer 2026

**Rôle de lecture :** encadrant académique ∩ recruteur AI Engineer ∩ reviewer technique RAG.
**Objet :** évaluation exclusive de l'existant. **Aucun nouveau développement n'est proposé.**

**Faits vérifiés (rappel des mesures d'audit réelles) :**

| Élément | Mesure |
|---|---|
| Chunking | 412+100=512 ; 2 026 chunks ; médiane 427 ; max 512 ; 0 tronqué ; 0 doublon ; overlap ≈ 98 ; 98 % frontières propres |
| Retrieval | Recall@5 = 75,0 % ; MRR = 72,8 % |
| Génération + Vérification | Faithfulness = 72,7 % ; Citation Accuracy = 80,0 % |
| Latences (CPU) | hybride 65-85 ms ; rerank 4-12 s ; Ollama 6-23 s ; gate NLI 1-2 s ; verify ≈ 1 s |
| Ablation NLI (même fenêtre) | verbatim : support 0,79/ent 0,82 ; paraphrase : support 0,26/neutral 0,72 ; contrefactuel : support 0,0001/contradiction 0,76 |
| Seuils | SUPPORT_LOW 0,40 ; SUPPORT_HIGH 0,70 ; GROUNDING 0,45 ; ANSWER_GROUNDING 0,15 ; PARAPHRASE_RESCUE 0,40 |
| Défauts tracés | 8 bugs (B1-B8), 7 correctifs, 1 résidu assumé |

---

## 1. Architecture

**Forces**
- Pipeline conforme à l'état de l'art : hybride dense+BM25 → RRF → cross-encodeur → génération ancrée → vérification NLI.
- `app.py` = coquille sans logique métier dupliquée (confinement réel).
- RRF (k=60) sans calibration entre échelles hétérogènes ; chunking contraint par le modèle (412+100=512 = borne BGE).

**Faiblesses**
- Deux étages NLI (gate interne + vérification affichée) avec prémisses différentes : coût doublé, incohérence de perception possible.
- SYSTEM_PROMPT surchargé pour un LLM de 3B, compensé par sanitation → complexité.

**Risques**
- Scripts de diagnostic accumulés dans `files/` : frontière production/labo poreuse.
- Évolution Ollama ou retrait du modèle → bascule silencieuse en Template.

**Note indicative : 8,5/10**

---

## 2. Génération

**Forces**
- « Question-first » + preuves, comportement constaté dans les traces.
- 6 étages de défense : parsing tolérant → sanitation → grounding → gate NLI → repli extractif → refus.
- Refus validé sur scénario déterministe (pool « sourdough »).

**Faiblesses**
- Volatilité réelle : à temp 0.2, le même prompt produit [1], [2,3], [1..5], de la prose, ou l'hallucination « real-world images ».
- Tolérance paraphrase lexicale (cov ≥ 0,40) : ne distingue pas paraphrase ancrée / fragment contaminé du pool (résidu documenté).
- Fallback extractif = verbatim, rompt la fluidité.

**Risques**
- Claim à ~52 % de vocabulaire partagé avec sa source citée peut passer avec une erreur sémantique. Contamination intra-pool non bloquée en amont.

**Note indicative : 7/10**

---

## 3. Retrieval

**Forces**
- Canaux réellement complémentaires ; RRF ; pool 20 puis top-5 cross-encoder ; paramètres contraints par le modèle ; reranker déterministe.

**Faiblesses**
- Résultats sans ablation chiffrée des canaux (contribution de chaque étage non quantifiée).
- Recall@5 + MRR seulement : pas de Recall@k∈{1,3,10}, pas de NDCG, pas d'analyse d'échecs.

**Risques**
- Score sur le top-5 injecté au LLM : biais possible selon critère d'annotation ; petit échantillon → écart-type de plusieurs points possibles.

**Note indicative : 7,5/10**
---

## 5. Évaluation

**Forces**
- Grille conforme RAGAS ; métriques rattachées à des traces rejouables ; écart Faithfulness/Citation Accuracy analysé.

**Faiblesses**
- Échantillon restreint, sans IC ni accord inter-annotateurs ; 3 chiffres significatifs trompeurs ; pas d'Answer/Context Relevancy, pas de taux de faux refus, pas de multi-runs.

**Risques**
- Faithfulness hérite du biais neutral du vérificateur → non indépendance possible de la métrique.

**Note indicative : 6/10**

---

## 6. Reproductibilité

**Forces**
- Artefacts : sorties brutes LLM, traces RAG_TRACE, hashes (sha256_of), rejeux déterministes. Retrieval/reranker/NLI déterministes.

**Faiblesses**
- Pas de seed Ollama, pas de multi-exécution, pas de verrou des versions, audit + production confondus.

**Risques**
- Un tiers reproduit le comportement, pas les nombres exacts ; dépendance Windows/modèles téléchargés.

**Note indicative : 7/10**

---

## 7. Qualité scientifique

**Forces**
- Preuve avant correctif (B1→B8) ; diagnostic différentiel hallucination/contamination/biais ; résidus consignés.

**Faiblesses**
- Fenêtre NLI par overlap lexical (ad hoc) ; seuils itérés non optimisés ; pas de bibliographie complète.

**Risques**
- Vérité terrain absente → boucle d'évaluation auto-référentielle.

**Note indicative : 6,5/10**

---

## 8. Qualité logicielle

**Forces**
- Modules découplés, dataclasses typées, compilation vérifiée, traçabilité STEP1→6, get_last_nli_scores, validations corpus.

**Faiblesses**
- Pas de pytest sur la prod ; ~30 scripts de diagnostic accumulés ; pas de linter/type-check/dev-deps visibles.

**Risques**
- Aucun garde-fou de régression auto ; friction d'adoption (téléchargement modèles non documenté).

**Note indicative : 6,5/10**


---

## 4. Citation Verification

**Forces**
- Vérification réellement câblée de bout en bout (segmentation → fenêtre locale → NLI → verdict affiché).
- Fenêtre locale = réponse à un problème mesuré (dilution sur 400-500 tokens).
- Ablation 0,79 vs 0,26 sur le même fait = preuve expérimentale du biais du vérifieur.

**Faiblesses**
- Seuils a priori non calibrés ; sensibilité élevée de la fenêtre (0,74 vs 0,07 selon source) ; table stricte parfois contradictoire avec le gate.

**Risques**
- « La métrique mesure ce que le vérificateur accepte » : preuve circulaire sans oracle externe.

**Note indicative : 8/10**
## 9. GitHub Readiness

**Forces**
- Contenu de démonstration fort ; histoire B1-B8 = storytelling README de premier ordre.

**Faiblesses / risques**
- Absence de README d'accueil, LICENSE, CI, structure src/tests, quarantaine des scripts de labo ; rejouabilité partielle (LLM) ; .gitignore à confirmer.

**Note indicative : 5,5/10**

---

## 10. Soutenance

**Forces**
- Histoire complète symptôme→instrumentation→preuve→patch→validation ; honnêteté ; exemples percutants (arXiv ID dans un JSON).

**Faiblesses / risques**
- Sur-narration sans preuves sous la main ; question « et si on change corpus/LLM/seuils ? » → arguments qualitatifs.

**Note indicative : 8/10**

---

## Grille de notation finale (sur 20)

| Contexte | Note | Justification |
|---|---|---|
| **PFE académique** | **15/20** | Ingénierie solide, audit exemplaire, limites assumées. Pénalisé par la rigueur scientifique (échantillon, seuils, oracle) et la structuration tests/repro. |
| **Portfolio AI Engineer** | **16/20** | Projet complet, démontrable, debugging crédible, métriques réelles — très différenciant. Pénalisé par hygiène de repo et variance de génération. |
| **Projet GitHub** | **13/20** | Excellent contenu mais finition éditoriale absente (README, licence, CI, séparation labo/prod). Potentiel 17-18 à coût marginal. |

**Synthèse forces :** traçabilité de bout en bout, défense anti-hallucination en profondeur, honnêteté scientifique, projet « racontable ».
**Synthèse faiblesses :** échantillon et calibration statistiques faibles, biais de l'évaluateur non décorrélé de la métrique, hygiène de dépôt, variance du générateur local.
---

## Les 5 questions les plus difficiles d'un jury / recruteur — et réponses attendues

**Q1 — « Votre Faithfulness (72,7 %) est calculée par un vérificateur dont vous avez prouvé qu'il classe la paraphrase "neutral". Ce chiffre mesure-t-il la qualité du système ou le biais de votre propre évaluateur ? Que vaut-il vraiment ? »**
Réponse attendue :
1. Reconnaître le biais, prouvé par ablation : le même fait passe de support 0,79 (verbatim) à 0,26 (paraphrase) sur la même fenêtre.
2. La Faithfulness affichée est une **borne basse conservatrice** (« supporté par vérificateur »), pas une vérité terrain.
3. Distinguer les trois classes : contradictions détectées de façon fiable (P≈0,76), paraphrases pénalisées par nature, inventaires hors contexte quasi impossibles.
4. La correction exigée est un **oracle externe** (annotation humaine ou LLM-juge) pour décorréler la métrique de l'implémentation.

**Q2 — « Votre gate accepte par tolérance lexicale un claim contenant la contamination "real-world images", que vous documentez vous-même. Concrètement, une réponse fausse peut être affichée. Pourquoi cela n'invalide-t-il pas la promesse anti-hallucination ? »**
Réponse attendue :
1. Le système est en couches ; la tolérance répond à un arbitrage mesuré : sans elle, des paraphrases correctes sont rejetées → remplacées par un assemblage hors sujet (pire pour l'utilisateur).
2. La contamination exige que la phrase fautive **existe exactement dans le pool** ET que le claim partage ≥ 40 % de vocabulaire avec sa source citée — résidu de type 2, minoritaire.
3. Gardes actives restantes : règle 5 du prompt (consistance citation-contenu), gate de contradiction, vérification finale stricte affichée.
4. Assumer le compromis false-accept/false-reject : zéro-risque exigerait un jury sémantique supplémentaire (hors périmètre).

**Q3 — « Le gate conserve des réponses que la table Citation Verification affiche à 0,01-0,16. Deux étages NLI, deux verdicts. Que doit croire l'utilisateur, et pourquoi ne partagez-vous pas un unique score ? »**
Réponse attendue :
1. Oui, deux évaluations coexistent (gate interne : prémisse phrase unique + tolérance ; table : fenêtre locale ±1 + seuils stricts).
2. Incohérence assumée : la table est volontairement conservatrice ; la réponse finale a été validée par un critère plus tolérant.
3. La réconciliation est prévue mais non branchée : `get_last_nli_scores()` expose les scores du gate pour que l'interface réutilise la même évaluation (≈2-3 s économisées) — contrainte de périmètre (app.py non modifié).
4. Règle de perception actuelle : « la table sous-estime, jamais ne surestime ».

**Q4 — « Recall@5 = 75 % : sur combien de questions, annotées par qui, quel accord inter-annotateurs ? Ce 75 % est-il mesuré sur le top-5 hybride ou après reranking, et quel est le gain chiffré du reranker ? »**
Réponse attendue :
1. Échantillon annoté restreint — valeur indicative, sans intervalle de confiance ni accord inter-annotateurs documenté (faiblesse assumée).
2. La métrique porte sur le **top-5 effectivement injecté au LLM**, donc **après reranking**.
3. Le gain du reranker est illustré qualitativement dans les traces (le rang du top hybride change), mais pas d'ablation chiffrée canal par canal — manque identifié.
4. Rediriger vers ce que le chiffre prouve (couverture du contexte) et ce qu'il ne prouve pas (optimalité de chaque étage).

**Q5 — « Le LLM est non déterministe (mesuré : [1] tantôt, [1..5] tantôt, prose tantôt JSON). Vos 75/72,8/72,7/80 viennent-ils d'un run unique ? Un tiers les reproduira-t-il ? »**
Réponse attendue :
1. Ce qui est déterministe : retrieval, RRF, reranking, NLI, sanitation, gate. Ce qui ne l'est pas : échantillonnage de la génération.
2. Les métriques de génération portent la variance du modèle — non quantifiée en multi-runs (faiblesse).
3. La rejouabilité décisionnelle est la garantie forte : sorties brutes archivées + traces → le pipeline fait exactement les mêmes choix.
4. Reproduire les nombres : température fixée à 0,2, modèle taggé, corpus hashé (sha256_of) ; un tiers reproduit la méthode et l'ordre de grandeur, pas les exacts décimaux. Proposition honnête : multi-runs et rapport moyenne ± écart-type.

---

*Fin de l'évaluation — audit exclusif de l'existant, aucun développement supplémentaire proposé.*
