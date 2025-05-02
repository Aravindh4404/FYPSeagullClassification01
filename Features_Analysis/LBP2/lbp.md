3. Methodology
3.1 Local Binary Pattern Analysis
The texture analysis methodology employs Local Binary Patterns (LBP) to quantify the textural characteristics of gull feather regions. LBP is a powerful texture descriptor that captures local spatial structure by comparing each pixel with its neighboring pixels.
3.1.1 LBP Computation
For a given pixel at position $(x_c, y_c)$ with intensity $g_c$, the LBP value is computed by comparing it with $P$ neighbors at radius $R$ using the following equation:
LBPP,R(xc,yc)=∑p=0P−1s(gp−gc)⋅2p\text{LBP}_{P,R}(x_c, y_c) = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^pLBPP,R​(xc​,yc​)=∑p=0P−1​s(gp​−gc​)⋅2p
where:

$g_p$ is the intensity of the neighbor pixel $p$
$s(x)$ is the step function defined as:

1, & \text{if } x \geq 0 \\
0, & \text{if } x < 0
\end{cases}$$

We implemented uniform LBP patterns, which account for transitions between 0s and 1s in the binary representation, thereby capturing fundamental texture elements like edges and corners while reducing dimensionality.

#### 3.1.2 Feature Extraction

From the LBP computation, three primary feature histograms were extracted:

1. **LBP Histogram**: Distribution of LBP codes across the region of interest
2. **Ones Histogram**: Distribution of the number of 1s in each LBP code, calculated as:
   $$H_{\text{ones}}(i) = \sum_{j=0}^{N-1} [b(j) = i]$$
   where $b(j)$ represents the number of 1s in pattern $j$, and $[·]$ is the Iverson bracket

3. **Transitions Histogram**: Distribution of 0-to-1 or 1-to-0 transitions in each LBP code:
   $$H_{\text{trans}}(i) = \sum_{j=0}^{N-1} [t(j) = i]$$
   where $t(j)$ counts the transitions in pattern $j$

### 3.2 Texture Feature Quantification

From these histograms, several texture properties were calculated to characterize the regions:

1. **Entropy**: Measures randomness in texture patterns
   $$E = -\sum_{i} h(i) \log_2 h(i)$$
   where $h(i)$ is the normalized histogram value at bin $i$

2. **Energy**: Measures uniformity of texture
   $$U = \sum_{i} h(i)^2$$

3. **Contrast**: Measures intensity variations
   $$C = \sum_{i} (i - \mu)^2 h(i)$$
   where $\mu$ is the mean value of the histogram

4. **Homogeneity**: Measures closeness of distribution
   $$H = \sum_{i} \frac{h(i)}{1 + |i - \mu|}$$

### 3.3 Distribution Comparison Metrics

To quantify differences between species, we employed the following statistical measures:

1. **Jensen-Shannon Distance**: Measure of similarity between probability distributions
   $$JSD(P||Q) = \sqrt{\frac{D_{KL}(P||M) + D_{KL}(Q||M)}{2}}$$
   where $M = \frac{P+Q}{2}$ and $D_{KL}$ is the Kullback-Leibler divergence

2. **Earth Mover's Distance**: Measures the minimum "work" required to transform one distribution into another
   $$EMD(P, Q) = \inf_{\gamma \in \Gamma(P, Q)} \int d(x, y) d\gamma(x, y)$$
   where $\Gamma(P, Q)$ is the set of all distributions with marginals $P$ and $Q$

3. **Chi-Square Distance**: Measures the goodness of fit between distributions
   $$\chi^2(P, Q) = \sum_{i} \frac{(P_i - Q_i)^2}{P_i + Q_i}$$

### 3.4 Dimensionality Reduction

Principal Component Analysis (PCA) was applied to visualize the high-dimensional LBP feature space in two dimensions:

$$\mathbf{X_{PCA}} = \mathbf{X} \cdot \mathbf{W}$$

where $\mathbf{W}$ is the matrix of eigenvectors of $\mathbf{X}^T\mathbf{X}$ corresponding to the two largest eigenvalues, and $\mathbf{X}$ is the standardized feature matrix.

This methodology allowed us to comprehensively analyze the textural differences between Slaty-backed and Glaucous-winged Gull feather patterns, providing both quantitative metrics and visual representations of the distinguishing characteristics.