# University Viva Defense Guide & Questions

**Project:** AI-Powered Resume & Job-Role Recommendation System  
**Reference Paper:** *AI-Powered Model for Intelligent Resume Recommendation and Feedback* (IEEE ICCCNT 2024)  
**Target Audience:** University External Examiners, Research Reviewers, and Viva Panels

---

## Part 14: MY 30-SECOND VIVA ANSWER

> "Our project is an AI-powered resume and job-role recommendation system that matches a candidate's resume to target software roles and provides structured skill gap feedback. We benchmarked against the IEEE ICCCNT 2024 reference paper which achieved 81.60% using BERT. Our initial baseline model used TF-IDF + SVD + Logistic Regression, achieving 66.40% cross-validation accuracy. We upgraded our representation to Sentence Transformers (`all-MiniLM-L6-v2`) + Logistic Regression because it captures contextual semantic meaning between technical skills. Our final model achieves **71.60% 5-Fold Cross-Validation Accuracy** (and **70.00% Out-Of-Fold Accuracy** with **69.19% Macro F1-score**) with zero data leakage. We also built an explainable skill-matching engine that pinpoints missing skills for the candidate."

---

## Part 13: MY 1-MINUTE PROJECT EXPLANATION

> **1. Problem:**  
> Traditional resume screening relies on simple keyword matching (like counting "Python" or "SQL"). This fails to understand the actual context of a candidate's experience (for example, distinguishing Python used for web backends versus Python used for Machine Learning).
> 
> **2. Research Paper:**  
> We followed the research paper *"AI-Powered Model for Intelligent Resume Recommendation and Feedback"* (IEEE ICCCNT 2024), which used BERT to solve this problem.
> 
> **3. Dataset:**  
> The paper used a general resume corpus (exact sample size not specified). Our project evaluated 100 verified industry job postings across 5 balanced roles: AI/ML, Full Stack, Software Engineer, Data Analyst, and Python Developer.
> 
> **4. Old Model & Limitation:**  
> Our initial baseline used TF-IDF with TruncatedSVD and Logistic Regression. It achieved **66.40%** cross-validated accuracy. It struggled because SVD dimensionality reduction caused information loss and could not disambiguate shared keywords across roles.
> 
> **5. New Model & Improvement:**  
> We investigated contextual representations and adopted **Sentence Transformers (`all-MiniLM-L6-v2`) with Logistic Regression**. This improved our cross-validated accuracy to **71.60%** ($+5.20\%$ absolute improvement) and Macro F1-score to **69.19%**.
> 
> **6. Comparison & Integrity:**  
> The paper reported 81.60% with full BERT fine-tuning on their dataset. Our 71.60% is rigorously evaluated using 5-Fold Repeated Stratified Cross-Validation on our 100-sample corpus with zero data leakage. We do not claim an artificial 81%+ because fine-tuning heavy BERT directly on 100 samples causes severe overfitting.
> 
> **7. Future Scope:**  
> Expanding the dataset to 1,000+ multi-industry job postings, supporting multi-page resume layouts, and integrating real-time automated interview question generation based on missing skills.

---

## Part 12: 25 Standard Viva Questions & Simple Answers

### Q1. What is your project?
**Answer:**  
"Our project is an intelligent resume recommendation and skill feedback system. It takes a candidate's resume PDF, extracts the text, uses semantic AI models to predict the best-matching job role out of 5 technical domains, calculates similarity scores, and highlights missing skills with career recommendations."

---

### Q2. What research paper did you follow?
**Answer:**  
"We followed the research paper titled: *'AI-Powered Model for Intelligent Resume Recommendation and Feedback'*, published in IEEE ICCCNT 2024 by Abhishek Mishra, Siddharth Singh, and Jisha R C."

---

### Q3. What dataset did the research paper use?
**Answer:**  
"The research paper used a dataset of technical resumes and job postings collected from online recruitment sources. The paper did not specify the exact dataset name or the exact sample count."

---

### Q4. What dataset are you using?
**Answer:**  
"We are using a dataset of 100 annotated job postings across 5 balanced technical categories (20 postings each): AIML, Full Stack, Software Engineer, Data Analyst, and Python jobs. The average posting length is 65.8 words."

---

### Q5. What model did the research paper use?
**Answer:**  
"The paper's primary and best-performing model was **BERT (`bert-base-uncased`)**, which uses bidirectional self-attention to generate contextual text embeddings."

---

### Q6. What other models did the paper compare?
**Answer:**  
"The paper compared four traditional machine learning baselines:
1. Linear SVM (66.2%)
2. Logistic Regression (65.9%)
3. Random Forest (65.2%)
4. Naive Bayes (53.3%)"

---

### Q7. What was the best accuracy in the paper?
**Answer:**  
"The best accuracy reported in the paper was **81.60%**, achieved by the BERT (`bert-base-uncased`) model."

---

### Q8. What was your old model?
**Answer:**  
"Our old baseline model was:  
**PDF Text $\rightarrow$ TF-IDF (1-2 n-grams) $\rightarrow$ TruncatedSVD (60 components) $\rightarrow$ L2 Normalizer $\rightarrow$ Logistic Regression ($C=1.5$).**"

---

### Q9. What was your old accuracy?
**Answer:**  
"Our old baseline achieved **66.40% Repeated Stratified CV accuracy** (and **67.00% Out-Of-Fold accuracy**) with a Macro F1-score of **65.67%**."

---

### Q10. What new model did you use?
**Answer:**  
"Our best new model is **Sentence Transformer (`sentence-transformers/all-MiniLM-L6-v2`) + Logistic Regression ($C=1.0$)**."

---

### Q11. Why did you change the model?
**Answer:**  
"We changed the model because TF-IDF only counts exact word frequencies and cannot understand word context or synonyms. For instance, 'Python developer building REST APIs' and 'Python developer training PyTorch neural networks' shared too many identical words under TF-IDF, confusing the model on Software Engineer and Python roles."

---

### Q12. Why is BERT/Transformer better for this problem?
**Answer:**  
"Transformers use self-attention mechanisms to understand the relationship between words in both directions. This allows the model to capture semantic context (e.g. knowing that 'PyTorch' relates to AI/ML and 'React' relates to Full Stack) even if common words like 'developer', 'experience', and 'years' appear in all resumes."

---

### Q13. What problem/limitation did you identify in the paper?
**Answer:**  
"We identified three key limitations:
1. The paper did not report Macro F1-score, Precision, Recall, or Confusion Matrices—only overall accuracy.
2. Fine-tuning a 110-million parameter BERT model is computationally expensive and slow for real-time web inference.
3. The paper did not explore lightweight Sentence Transformers or fine-grained skill gap extraction."

---

### Q14. How did your project improve it?
**Answer:**  
"1. We implemented lightweight **Sentence Transformers (MiniLM-L6)** which run $>5\times$ faster on standard CPUs.  
2. We conducted a strict 11-model benchmark with zero data leakage, reporting full Precision, Recall, Macro F1, and Confusion Matrices.  
3. We integrated a deterministic skill-gap engine that extracts matched and missing skills for candidate feedback."

---

### Q15. What is the accuracy of your final model?
**Answer:**  
"Our final model achieves **71.60% 5-Fold Cross-Validation Accuracy** ($\pm 10.17\%$), **70.00% Out-Of-Fold Accuracy**, **68.92% Macro Precision**, **70.00% Macro Recall**, and **69.19% Macro F1-Score**."

---

### Q16. Is your model better than the paper?
**Answer:**  
"Our model outperforms our previous baseline by **$+5.20$ percentage points** (71.60% vs 66.40%). However, compared to the paper's reported 81.60%, our accuracy is 71.60% because our evaluation was conducted on our 100-sample dataset using strict 5-fold cross-validation. Direct numerical comparison across different datasets is not scientifically valid, as stated in our comparative report."

---

### Q17. How did you make sure your accuracy is genuine?
**Answer:**  
"1. We used **5-Fold Repeated Stratified Cross-Validation (25 splits)** so every sample was evaluated on an unseen test fold.  
2. All preprocessing and transformations were fit strictly inside the training fold to eliminate train/test leakage.  
3. We never evaluated on the training set or manipulated labels. All metrics are backed by out-of-fold predictions."

---

### Q18. What is overfitting?
**Answer:**  
"Overfitting occurs when a machine learning model memorizes the training data (including noise and specific phrasing) rather than learning general patterns. An overfitted model gets near 100% training accuracy but performs poorly on new, unseen resumes."

---

### Q19. How did you prevent overfitting?
**Answer:**  
"1. We used a **pre-trained frozen Sentence Transformer** encoder (`all-MiniLM-L6-v2`) rather than fine-tuning 110M parameters on 100 samples.  
2. We applied L2-regularization ($C=1.0$) on the Logistic Regression classifier head.  
3. We evaluated using Stratified Cross-Validation across multiple seeds."

---

### Q20. What is the difference between accuracy, precision, recall and F1?
**Answer:**  
- **Accuracy:** The overall percentage of correctly classified resumes out of all resumes ($\frac{TP+TN}{\text{Total}}$).  
- **Precision:** When the model predicts 'AIML', how many are actually 'AIML' ($\frac{TP}{TP+FP}$).  
- **Recall:** Out of all actual 'AIML' postings, how many did the model correctly identify ($\frac{TP}{TP+FN}$).  
- **F1-Score:** The harmonic mean of precision and recall ($2 \cdot \frac{P \cdot R}{P + R}$), which gives a balanced measure of performance on each class.

---

### Q21. Why did you choose your final model?
**Answer:**  
"We selected **SBERT (MiniLM-L6) + Logistic Regression** because it achieved the highest cross-validated accuracy (**71.60%**) and highest Macro F1-score (**69.19%**) while maintaining ultra-fast inference speed (~30ms per resume) on standard CPU hardware without GPU requirements."

---

### Q22. What happens if we use Random Forest instead?
**Answer:**  
"In the reference paper, Random Forest scored **65.2%**, lower than Linear SVM and Logistic Regression. In our small 100-sample high-dimensional space, tree-based models tend to overfit on specific keyword thresholds and lack the smooth linear separation boundaries of regularized linear models."

---

### Q23. What happens if we use SVM instead?
**Answer:**  
"Linear SVM performs well on sparse text (paper reported 66.2%; our SBERT + LinearSVC achieved 68.4% CV). However, Logistic Regression yields calibrated probability distributions (softmax), which are essential for showing the candidate their percentage match across all 5 roles rather than just an uncalibrated margin score."

---

### Q24. What happens if we use BERT instead of TF-IDF?
**Answer:**  
"Using BERT / Sentence Transformers replaces sparse orthogonal word counts with dense contextual embeddings. It maps synonyms and related skills (e.g. 'PyTorch' and 'Deep Learning') to nearby points in embedding space, raising classification accuracy from ~66% to ~71.6% and dramatically reducing confusion on overlapping roles."

---

### Q25. What is the future scope?
**Answer:**  
"1. Expanding the job database to cover 50+ diverse domains.  
2. Parsing complex multi-column and graphical resume formats with OCR/LLM vision parsers.  
3. Implementing personalized interview preparation generation tailored to the candidate's identified skill gaps."
