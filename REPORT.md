# **Word Embeddings**

### Pre-Trained Models
```html
https://iiitaphyd-my.sharepoint.com/:f:/g/personal/harsh_bansal_students_iiit_ac_in/EiGhi1L_jwFMl6WMZ1OgvcUBGLXgYTbePes0HSjzRD-1Uw?e=JvCzaU
```

---

### **Note**

- Both the Skip-Gram and SVD Vectorizations are run on `40000` Sentences due to Computational Constraints.
- Used `punkt` tokenizer from `nltk` for tokenizing the Sentences.

---

# **SVD Vectorization**

### **Assumptions**

- Use `lil_matrix` and `csr_matrix` from `scipy.sparse` for efficient computation of co-occurence matrix
- Used `randomized_svd` from `sklearn.utils.extmath` for faster svd computation.

---

### **Hyper-Parameters**

```
- Embedding Dimension: 300
- Context Window Size: 1
```

---

### **Hyper-Parameters for LSTM**

```
- Hidden Size: 256
- Learning Rate: 0.001
- Number of Epochs: 25
- Batch Size: 100
```

---

### **Results for DownStream Tasks**

> Train Dataset Metrics

```
Accuracy: 0.8386
F1 Score: 0.8378667489017662
Precision: 0.8393609446795577
Recall: 0.8386
Confusion Matrix:
[[9112  474  354  621]
 [ 269 9185   87  356]
 [ 606  282 6968 1522]
 [ 630  445  810 8279]]
```

> Test Dataset Metrics

```
Accuracy: 0.8048684210526316
F1 Score: 0.8039164979378661
Precision: 0.8057255133407202
Recall: 0.8048684210526316
Confusion Matrix:
[[1593  100   76  131]
 [  72 1731   28   69]
 [ 126   63 1346  365]
 [ 131  129  193 1447]]
```

---

# **Skip-Gram Vectorization**

### **Assumptions**

- Cutoff Frequency: 10
- Number of Negative Samples per Positive Sample: 2

---

### **Hyper-Parameters of Generating Skip-Gram Embeddings**

```
- Embedding Dimension: 300
- Context Window Size: [1, 2, 3, 4, 5]
- Hidden Size: 256
- Learning Rate: 0.1
- Number of Epochs: 10
- Batch Size: 256
```

---

### **Results for Multiple Context Window Size**

> Context Window: 1

```
Generating Positive & Negative Samples: 100%|██████████| 40000/40000 [00:29<00:00, 1334.27it/s]
Context Window: 1, Epoch 1/10: 100%|██████████| 11016/11016 [03:56<00:00, 46.66batch/s, loss=0.0748]
Context Window: 1, Epoch: 1, Loss: 0.0748
Context Window: 1, Epoch 2/10: 100%|██████████| 11016/11016 [03:55<00:00, 46.79batch/s, loss=0.0268]
Context Window: 1, Epoch: 2, Loss: 0.0268
Context Window: 1, Epoch 3/10: 100%|██████████| 11016/11016 [03:55<00:00, 46.81batch/s, loss=0.016]
Context Window: 1, Epoch: 3, Loss: 0.0160
Context Window: 1, Epoch 4/10: 100%|██████████| 11016/11016 [03:56<00:00, 46.68batch/s, loss=0.0114]
Context Window: 1, Epoch: 4, Loss: 0.0114
Context Window: 1, Epoch 5/10: 100%|██████████| 11016/11016 [03:55<00:00, 46.80batch/s, loss=0.00912]
Context Window: 1, Epoch: 5, Loss: 0.0091
Context Window: 1, Epoch 6/10: 100%|██████████| 11016/11016 [03:55<00:00, 46.72batch/s, loss=0.00786]
Context Window: 1, Epoch: 6, Loss: 0.0079
Context Window: 1, Epoch 7/10: 100%|██████████| 11016/11016 [03:55<00:00, 46.87batch/s, loss=0.00715]
Context Window: 1, Epoch: 7, Loss: 0.0071
Context Window: 1, Epoch 8/10: 100%|██████████| 11016/11016 [03:55<00:00, 46.87batch/s, loss=0.0067]
Context Window: 1, Epoch: 8, Loss: 0.0067
Context Window: 1, Epoch 9/10: 100%|██████████| 11016/11016 [03:55<00:00, 46.86batch/s, loss=0.00641]
Context Window: 1, Epoch: 9, Loss: 0.0064
Context Window: 1, Epoch 10/10: 100%|██████████| 11016/11016 [03:55<00:00, 46.86batch/s, loss=0.00621]
Context Window: 1, Epoch: 10, Loss: 0.0062
```

> Context Window: 2

```
Generating Positive & Negative Samples: 100%|██████████| 40000/40000 [00:55<00:00, 726.28it/s]
Context Window: 2, Epoch 1/10: 100%|██████████| 21719/21719 [07:42<00:00, 46.91batch/s, loss=0.0582]
Context Window: 2, Epoch: 1, Loss: 0.0582
Context Window: 2, Epoch 2/10: 100%|██████████| 21719/21719 [07:45<00:00, 46.67batch/s, loss=0.0209]
Context Window: 2, Epoch: 2, Loss: 0.0209
Context Window: 2, Epoch 3/10: 100%|██████████| 21719/21719 [07:43<00:00, 46.87batch/s, loss=0.0141]
Context Window: 2, Epoch: 3, Loss: 0.0141
Context Window: 2, Epoch 4/10: 100%|██████████| 21719/21719 [07:43<00:00, 46.81batch/s, loss=0.0111]
Context Window: 2, Epoch: 4, Loss: 0.0111
Context Window: 2, Epoch 5/10: 100%|██████████| 21719/21719 [07:44<00:00, 46.80batch/s, loss=0.00957]
Context Window: 2, Epoch: 5, Loss: 0.0096
Context Window: 2, Epoch 6/10: 100%|██████████| 21719/21719 [07:44<00:00, 46.74batch/s, loss=0.00868]
Context Window: 2, Epoch: 6, Loss: 0.0087
Context Window: 2, Epoch 7/10: 100%|██████████| 21719/21719 [07:43<00:00, 46.82batch/s, loss=0.00813]
Context Window: 2, Epoch: 7, Loss: 0.0081
Context Window: 2, Epoch 8/10: 100%|██████████| 21719/21719 [07:44<00:00, 46.75batch/s, loss=0.00776]
Context Window: 2, Epoch: 8, Loss: 0.0078
Context Window: 2, Epoch 9/10: 100%|██████████| 21719/21719 [07:43<00:00, 46.88batch/s, loss=0.00752]
Context Window: 2, Epoch: 9, Loss: 0.0075
Context Window: 2, Epoch 10/10: 100%|██████████| 21719/21719 [07:42<00:00, 46.92batch/s, loss=0.00734]
Context Window: 2, Epoch: 10, Loss: 0.0073
```

> Context Window: 3

```
Generating Positive & Negative Samples: 100%|██████████| 40000/40000 [01:27<00:00, 455.44it/s]
Context Window: 3, Epoch 1/10: 100%|██████████| 32110/32110 [12:29<00:00, 42.83batch/s, loss=0.05]
Context Window: 3, Epoch: 1, Loss: 0.0500
Context Window: 3, Epoch 2/10: 100%|██████████| 32110/32110 [12:41<00:00, 42.16batch/s, loss=0.0186]
Context Window: 3, Epoch: 2, Loss: 0.0186
Context Window: 3, Epoch 3/10: 100%|██████████| 32110/32110 [12:32<00:00, 42.66batch/s, loss=0.0132]
Context Window: 3, Epoch: 3, Loss: 0.0132
Context Window: 3, Epoch 4/10: 100%|██████████| 32110/32110 [12:33<00:00, 42.62batch/s, loss=0.0109]
Context Window: 3, Epoch: 4, Loss: 0.0109
Context Window: 3, Epoch 5/10: 100%|██████████| 32110/32110 [12:42<00:00, 42.13batch/s, loss=0.0097]
Context Window: 3, Epoch: 5, Loss: 0.0097
Context Window: 3, Epoch 6/10: 100%|██████████| 32110/32110 [12:46<00:00, 41.88batch/s, loss=0.00898]
Context Window: 3, Epoch: 6, Loss: 0.0090
Context Window: 3, Epoch 7/10: 100%|██████████| 32110/32110 [12:46<00:00, 41.87batch/s, loss=0.00852]
Context Window: 3, Epoch: 7, Loss: 0.0085
Context Window: 3, Epoch 8/10: 100%|██████████| 32110/32110 [12:50<00:00, 41.68batch/s, loss=0.00822]
Context Window: 3, Epoch: 8, Loss: 0.0082
Context Window: 3, Epoch 9/10: 100%|██████████| 32110/32110 [12:45<00:00, 41.97batch/s, loss=0.008]
Context Window: 3, Epoch: 9, Loss: 0.0080
Context Window: 3, Epoch 10/10: 100%|██████████| 32110/32110 [12:42<00:00, 42.12batch/s, loss=0.00783]
Context Window: 3, Epoch: 10, Loss: 0.0078
```

> Context Window: 4

```
Generating Positive & Negative Samples: 100%|██████████| 40000/40000 [01:52<00:00, 354.98it/s]
Context Window: 4, Epoch 1/10: 100%|██████████| 42188/42188 [16:42<00:00, 42.08batch/s, loss=0.0446]
Context Window: 4, Epoch: 1, Loss: 0.0446
Context Window: 4, Epoch 2/10: 100%|██████████| 42188/42188 [16:40<00:00, 42.16batch/s, loss=0.0171]
Context Window: 4, Epoch: 2, Loss: 0.0171
Context Window: 4, Epoch 3/10: 100%|██████████| 42188/42188 [16:41<00:00, 42.13batch/s, loss=0.0126]
Context Window: 4, Epoch: 3, Loss: 0.0126
Context Window: 4, Epoch 4/10: 100%|██████████| 42188/42188 [16:38<00:00, 42.25batch/s, loss=0.0107]
Context Window: 4, Epoch: 4, Loss: 0.0107
Context Window: 4, Epoch 5/10: 100%|██████████| 42188/42188 [16:36<00:00, 42.32batch/s, loss=0.00972]
Context Window: 4, Epoch: 5, Loss: 0.0097
Context Window: 4, Epoch 6/10: 100%|██████████| 42188/42188 [16:36<00:00, 42.34batch/s, loss=0.00911]
Context Window: 4, Epoch: 6, Loss: 0.0091
Context Window: 4, Epoch 7/10: 100%|██████████| 42188/42188 [16:39<00:00, 42.20batch/s, loss=0.00871]
Context Window: 4, Epoch: 7, Loss: 0.0087
Context Window: 4, Epoch 8/10: 100%|██████████| 42188/42188 [16:43<00:00, 42.05batch/s, loss=0.00844]
Context Window: 4, Epoch: 8, Loss: 0.0084
Context Window: 4, Epoch 9/10: 100%|██████████| 42188/42188 [16:46<00:00, 41.92batch/s, loss=0.00825]
Context Window: 4, Epoch: 9, Loss: 0.0083
Context Window: 4, Epoch 10/10: 100%|██████████| 42188/42188 [16:46<00:00, 41.91batch/s, loss=0.0081]
Context Window: 4, Epoch: 10, Loss: 0.0081
```

> Context Window: 5

```
Generating Positive & Negative Samples: 100%|██████████| 40000/40000 [02:17<00:00, 290.79it/s]
Context Window: 5, Epoch 1/10: 100%|██████████| 51953/51953 [20:39<00:00, 41.90batch/s, loss=0.0408]
Context Window: 5, Epoch: 1, Loss: 0.0408
Context Window: 5, Epoch 2/10: 100%|██████████| 51953/51953 [20:30<00:00, 42.23batch/s, loss=0.0161]
Context Window: 5, Epoch: 2, Loss: 0.0161
Context Window: 5, Epoch 3/10: 100%|██████████| 51953/51953 [20:27<00:00, 42.32batch/s, loss=0.0122]
Context Window: 5, Epoch: 3, Loss: 0.0122
Context Window: 5, Epoch 4/10: 100%|██████████| 51953/51953 [20:27<00:00, 42.31batch/s, loss=0.0106]
Context Window: 5, Epoch: 4, Loss: 0.0106
Context Window: 5, Epoch 5/10: 100%|██████████| 51953/51953 [20:25<00:00, 42.41batch/s, loss=0.00969]
Context Window: 5, Epoch: 5, Loss: 0.0097
Context Window: 5, Epoch 6/10: 100%|██████████| 51953/51953 [20:26<00:00, 42.36batch/s, loss=0.00917]
Context Window: 5, Epoch: 6, Loss: 0.0092
Context Window: 5, Epoch 7/10: 100%|██████████| 51953/51953 [20:23<00:00, 42.45batch/s, loss=0.00882]
Context Window: 5, Epoch: 7, Loss: 0.0088
Context Window: 5, Epoch 8/10: 100%|██████████| 51953/51953 [20:21<00:00, 42.54batch/s, loss=0.00858]
Context Window: 5, Epoch: 8, Loss: 0.0086
Context Window: 5, Epoch 9/10: 100%|██████████| 51953/51953 [20:18<00:00, 42.65batch/s, loss=0.00841]
Context Window: 5, Epoch: 9, Loss: 0.0084
Context Window: 5, Epoch 10/10: 100%|██████████| 51953/51953 [20:17<00:00, 42.68batch/s, loss=0.00828]
Context Window: 5, Epoch: 10, Loss: 0.0083
```

Context window 3 means considering 3 next and 3 previous words, totaling 7 words. For a context window of 4, it is 9, and for 5, it is 11. During training, we observe that context window sizes 2 and 3 perform similarly, while size 1 consistently outperforms all. This is because a smaller context window captures more precise semantic relationships between words without introducing excessive noise from distant words.

---

### **Results for DownStream Tasks**

> Train Dataset Metrics

```
Accuracy: 0.8903722593064827
F1 Score: 0.8900546313696643
Precision: 0.8915315986504361
Recall: 0.8903722593064827
Confusion Matrix:
[[9244  435  372  510]
 [ 186 9514   55  142]
 [ 321  183 7733 1140]
 [ 318  254  469 9123]]
```

> Test Dataset Metrics

```
Accuracy: 0.8398684210526316
F1 Score: 0.839065590449206
Precision: 0.840891304604194
Recall: 0.8398684210526316
Confusion Matrix:
[[1592   86  100  122]
 [  48 1792   29   31]
 [ 104   57 1412  327]
 [  81   82  150 1587]]
```

---

### SVD vs. Skipgram Word Embeddings

#### Performance Comparison:

1. **SVD Word Embeddings:**

   - **Accuracy:** 0.8049
   - **F1 Score:** 0.8039
   - **Precision:** 0.8057
   - **Recall:** 0.8049

2. **Skipgram Word Embeddings:**
   - **Accuracy:** 0.8399
   - **F1 Score:** 0.8391
   - **Precision:** 0.8409
   - **Recall:** 0.8399

#### Analysis:

1. **Contextual Understanding:**

   - **SVD:** Singular Value Decomposition (SVD) based word embeddings typically capture co-occurrence statistics of words in a corpus. However, they might not effectively capture contextual nuances, as they treat each word pair equally regardless of their proximity or importance in the sentence.
   - **Skipgram:** Skipgram, on the other hand, is a neural network-based model, which learns to predict the context (surrounding words) given a target word. It captures more nuanced relationships between words, leveraging the surrounding context to better understand the meaning of each word.

2. **Data Efficiency:**

   - **SVD:** SVD requires processing the entire co-occurrence matrix, which can be computationally expensive and memory-intensive, especially for large corpora. It might struggle to scale efficiently to very large datasets.
   - **Skipgram:** Skipgram uses a shallow neural network architecture, making it more scalable and efficient for large datasets. It can process data in batches, enabling faster training on large corpora.

3. **Representation Power:**
   - **SVD:** SVD generates dense embeddings that might struggle to capture complex semantic relationships, especially in high-dimensional spaces. It might not effectively represent rare or polysemous words due to its linear projection approach.
   - **Skipgram:** Skipgram generates embeddings that tend to capture more nuanced semantic relationships and can handle polysemy better. It can learn complex patterns and capture semantic similarities more effectively.

#### Possible Shortcomings:

1. **SVD:**

   - **Dimensionality Reduction:** SVD often requires dimensionality reduction techniques to handle the high dimensionality of word co-occurrence matrices, which might lead to information loss.
   - **Lack of Contextual Information:** SVD ignores the sequential nature of language and treats all word pairs equally, potentially missing out on crucial contextual information.

2. **Skipgram:**
   - **Data Dependency:** Skipgram requires a large amount of training data to learn effective embeddings, which might be a limitation for resource-constrained environments.
   - **Context Window Size Sensitivity**: Skip-gram relies on defining a context window size, which determines the neighboring words used to predict the target word. This parameter can significantly impact the quality of embeddings, and finding the optimal window size might require experimentation. SVD doesn't have this sensitivity to context window size since it operates directly on the co-occurrence matrix.
   - **Training Complexity:** Training Skipgram models can be computationally intensive, especially for large vocabularies, and may require substantial computational resources.

#### Conclusion:

In this comparison, Skipgram outperforms SVD in terms of performance metrics, indicating its superiority in capturing semantic relationships and contextual nuances. Skipgram's ability to learn from context and its flexibility in representing complex semantic structures contribute to its better performance.
