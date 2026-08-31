% =============================================================================
% First-year PhD progression review -- merged document
% Part I : Topological Feature Learning for Parameter Estimation in
%          Neyman-Scott Processes (ParamNet)
% Part II: Hessian Eigenvalue Spectra During Grokking
% Flavio Gualtieri
% =============================================================================

\documentclass[11pt,a4paper]{article}

% --- geometry reconciled to a single setting for the merged document --------
\usepackage[margin=2.0cm]{geometry}
\usepackage{mathpazo}
\usepackage{microtype}

\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsthm}

\usepackage{graphicx}
\graphicspath{{../figs/}}
\usepackage{booktabs}
\usepackage{enumitem}
\usepackage[font=small,labelfont=bf,skip=4pt]{caption}
\usepackage{subcaption}
\usepackage{titlesec}
\titlespacing*{\section}{0pt}{1.0ex plus .2ex}{0.5ex plus .1ex}
\titlespacing*{\subsection}{0pt}{0.9ex plus .2ex}{0.4ex plus .1ex}
\titlespacing*{\paragraph}{0pt}{0.6ex plus .1ex}{1em}
\setlength{\tabcolsep}{5pt}
\renewcommand{\arraystretch}{1.0}
\setlength{\abovedisplayskip}{5pt}
\setlength{\belowdisplayskip}{5pt}
\setlength{\abovedisplayshortskip}{3pt}
\setlength{\belowdisplayshortskip}{3pt}

\usepackage{natbib}
\usepackage[hidelinks]{hyperref}
\raggedbottom
\renewcommand{\textfraction}{0.05}
\renewcommand{\topfraction}{0.95}
\renewcommand{\bottomfraction}{0.95}
\renewcommand{\floatpagefraction}{0.85}

% ----------------------------------------------------------------------------
% Macros
% ----------------------------------------------------------------------------
\newcommand{\R}{\mathbb{R}}
\newcommand{\N}{\mathbb{N}}
\newcommand{\Z}{\mathbb{Z}}
\newcommand{\E}{\mathbb{E}}
\newcommand{\pc}{\mathbf{x}}
\newcommand{\simpcom}{\Sigma}
\newcommand{\filt}{\mathcal{F}}
\newcommand{\kpar}{\kappa}
\newcommand{\mo}{\mu}
\newcommand{\bH}{\mathbf{H}}

% Visible placeholder box for figures that are not yet produced
\newcommand{\figtodo}[1]{\par\vspace{4pt}\noindent\fbox{\parbox{0.97\linewidth}%
  {\footnotesize\textbf{TODO (figure):} #1}}\par\vspace{4pt}}

% ----------------------------------------------------------------------------
\title{First-Year Progression Review\\[4pt]
       \large Topological Feature Learning for Parameter Estimation in
       Neyman-Scott Processes; Hessian Spectra During Grokking}
\author{Flavio Gualtieri}
\date{\today}

\begin{document}
\maketitle

\noindent
This review reports two disjoint projects from my first year. The first project -- the most recent, and considerably more substantial -- is a study of persistence-based feature learning for parameter estimation in Neyman-Scott point processes. This consists of a synthetic point process simulator, the ParamNet estimator and its
architectural ablations, and a comparison of topological vectorizations. The second -- largely exploratory -- concerns the dynamics of a neural network's Hessian eigenvalue spectrum throughout training, and how it relates to the phenomenon of Grokking.

% #############################################################################
\part{Topological Feature Learning for Parameter Estimation in Neyman-Scott
Processes}
% #############################################################################

% #############################################################################
\section{Introduction}
\label{sec:intro}
% #############################################################################

The study of spatial point processes (SPPs) finds applications in astronomy, forest ecology, epideiology, etc. Many natural phenomena are modelled using SPPs, for example \citet{neyman1952theory} model the spatial distribution of galaxies as a clustered point process;
\citet{owen2010palm} treat super-resolution microscopy localisations of membrane proteins as a point pattern and quantify their clustering with Ripley's $K$, and \citet{diggle2005surveillance} model individual disease cases as a spatio-temporal point process for on-line surveillance; see
\citet{moller2004statistical} for a general treatment.

The parameters of spatial point processes are of particular interest because they are interpretable scientific quantities. For example, when Neyman-Scott processes are used to model galaxy clustering the parent intensity $\kappa$ corresponds to the number density of haloes hosting galaxies, the offspring intensity $\mu$ is the mean occupancy of a halo, and the cluster scale $\sigma$ is the spatial extent of the galaxy distribution within a halo (\citet{yip2025learning}).

Existing estimation methods typically rely on closed-form expressions for
summary statistics such as the $K$ and $L$ functions. Minimum contrast
estimation fits parameters by numerically minimizing a discrepancy between the empirical and theoretical curve, most commonly for $K$ or $g$
\citep{diggle2003spatial}. Palm likelihood instead builds a composite likelihood from the process's Palm intensity, developed specifically for Neyman-Scott processes \citep{tanaka2008parameter}.

These methods share three limitations. Firstly, they rely on closed-form summary statistics, which may not exist: no closed-form $K$ has been derived for the nested Thomas process considered in this work. They may also exist but fail the regularity conditions the estimator's theory needs (the two-step estimator of \citet{waagepetersen2009twostep} requires $K_\psi$ to be twice continuously differentiable in $\psi$, which excludes the Mat\'ern cluster process). Secondly, they are sensitive to hyperparameters chosen per dataset by rule of thumb: minimum contrast depends on $(r_{max}, q, p)$, and \citet{waagepetersen2009twostep} mention that Thomas estimates of $\kappa$ and $\omega$ are ``quite sensitive'' to $r_{max}$ and $q$, being strongly negatively correlated so that the product $\hat\kappa\hat\omega$ is stable while the individual estimates are not. Finally,
they destabilise as the process approaches complete spatial randomness: writing $\lambda=\kappa\mu$ for the expected number of points per unit area and $\nu = \sigma\sqrt{\kappa}$ for the ratio of cluster radius to mean parent spacing, once $\nu \gtrsim 1$ neighbouring clusters overlap and many
$(\kappa,\mu)$ pairs with the same product produce nearly the same observable
pattern (Figure~\ref{fig:regime-grid}). This can be seen explicitly in the theoretical $K$-function; for a Thomas
process we have $K(t) = \pi t^2 + (1-e^{-t^2/4\sigma^2})/\kappa$, so as $\sigma$ grows at fixed $t$ the second term vanishes and $K$ tends to the Poisson $K$
regardless of $\kappa$.

\begin{figure}[ht]
\centering
\includegraphics[width=0.97\linewidth]{regime_grid.pdf}
\caption{Simulated realisations across the overlap index
$\nu \in \{0.1,0.3,0.6,0.9\}$ at fixed $\lambda,\kappa$: Thomas (top) vs.\
nested Thomas (bottom). One seed per panel, generated with the method explained in Section~\ref{ssec:simulator}. We can see in the top-right panel that the Thomas process approaches Poisson as $\nu$ increases.}
\label{fig:regime-grid}
\end{figure}

These limitations prompted me to consider whether tools from topological data analysis could be applied in this context, and whether they may have advantages over the established methods. In particular, persistence diagrams provide a multi-scale, process-agnostic descriptor of a point cloud that requires neither a closed form nor a choice of summary statistic.

% -----------------------------------------------------------------------------
\subsection{Problem setup}
\label{ssec:setup}
% -----------------------------------------------------------------------------

Let $X$ be a Neyman-Scott process on $\R^d$ with unknown parameter vector
$\theta \in \R^{n_{params}}$. For example, the homogeneous Thomas process has
$\theta = (\kappa, \mu, \sigma)$ corresponding to parent intensity, offspring
intensity and cluster scale, respectively. We observe a single realization of
$X$ on a bounded window $W \subset \R^d$, giving a point cloud
$\pc = \{x_1, \dots, x_n\} \subset \R^d$ with $N(\pc) = |X \cap W|$. The objective is
to produce an estimate $\hat\theta = \hat\theta(\pc)$ of $\theta$ from $\pc$
alone.

The model is trained on a set of synthetic clouds
$\{\pc_i, \theta_i\}_{i=1}^{N_{train}}$ by backpropagating the RMS loss
    \[
    \mathcal{L}(\hat\theta_i, \theta_i) = \sqrt{\sum_{j=1}^{n_{params}}(\hat\theta^j_i - \theta^j_i)^2}.
    \]
The scale of each parameter varies; therefore -- if not normalized -- the
large-scale parameters will dominate this loss. To address this, we train in
log-normalized parameter space, mapping
$\theta_i \mapsto (\log(\theta_i) - \mu_{\log\theta})/\sigma_{\log\theta}$,
where -- abusing notation -- $\mu_{\log\theta}, \sigma_{\log\theta}$ denote the
mean and standard deviation of the logged parameters over the training set. All
losses reported below are in this space, so that $\mathcal{L}\approx 1$
corresponds to chance level.

\newpage

% #############################################################################
\section{Background}
\label{sec:background}
% #############################################################################

\citet{vihrs2022neural} first proposed the use of deep learning for parameter
estimation. \citet{yip2025learning} employ a similar architecture with
persistence images as the input feature, mapping $H_0$--$H_2$ images of an
$\alpha\mathrm{DTM}\ell$ filtration of a halo catalogue to
$(\Omega_m,\sigma_8)$; their targets are cosmological parameters rather than
those of the underlying point process, and they do not systematically compare
topological summaries. 

The choice of topological summary (and vectorization method thereof) has been the central question of this investigation. Since persistence diagrams do not live in a Hilbert space, they cannot be fed directly to a machine learning models. The vectorization of persistence diagrams is an active area of research: \citet{adams2017persistence} propose persistence
images, \citet{umeda2017time} Betti curves, \citet{bubenik2015landscapes}
persistence landscapes and \citet{chazal2014silhouette} silhouettes. In this project I have experimented with each of these vectorization methods, and have found that -- although there is certainly room for methodological improvement -- the persistence image vectorization is the most promising candidate.

% -----------------------------------------------------------------------------
\subsection{Neyman-Scott processes}
\label{ssec:spp}
% -----------------------------------------------------------------------------

A spatial point process on $W \subseteq \R^d$ is characterised by its intensity
function $\rho(u)$, which satisfies $\E[N(W)] = \int_W \rho(u)\,du$ for
$N(W) = |W \cap X|$; when $\rho$ is constant the process is homogeneous, the
canonical example being the Poisson process. A Neyman-Scott process is a
clustered process built on top of a Poisson; the processes used throughout this work follow the standard construction:
\begin{enumerate}
    \item Expand $W$ by an edge buffer $b$ to produce $W_{exp}$, so that
    clusters whose parents lie outside $W$ are not truncated.
    \item Sample the parent process on $W_{exp}$:
    $n \sim \mathrm{Poisson}(\kappa|W_{exp}|)$, with parents placed uniformly
    on $W_{exp}$.
    \item For each parent $p$ independently, draw an offspring count
    $n_p \sim \mathrm{Poisson}(\mu)$.
    \item For each offspring independently, draw a displacement
    $\epsilon \sim k$ and place the point at $p + \epsilon$.
    \item Discard the parents and retain the offspring lying in $W$.
\end{enumerate}

Every Neyman-Scott process therefore has the parent and offspring intensities
$(\kappa,\mu)$ as parameters, and the choice of kernel supplies the rest: for
the \emph{Thomas} process $k = \mathcal{N}(0,\sigma^2 I_d)$, giving
$\theta = (\kappa,\mu,\sigma)$; for the Mat\'ern cluster process $k$ is uniform
on a ball of radius $R$; and for the \emph{nested Thomas} process the parent
process at the second step is itself a Thomas process rather than a Poisson one,
producing clusters of clusters and a five-parameter target $\theta = (\kappa_{meta},\mu_{meta},\kappa,\mu, \sigma)$.

% -----------------------------------------------------------------------------
\subsection{Persistent homology}
\label{ssec:persistence}
% -----------------------------------------------------------------------------
Persistent homology (PH)  tracks a growing family of simplicial complexes built on a point cloud. A
\emph{filtration function} $f$ assigns to each subset of points a scale at
which it appears in the resulting filtration $\filt_r$, subject to a monotonicity condition, and the sublevel sets
$\filt_r = \{\sigma : f(\sigma)\le r\}$ form a nested family of complexes
indexed by $r$. The canonical choice is Vietoris-Rips, where a simplex enters at
its diameter; the DTM$_k$ filtration \citep{anai2020dtm} instead uses each
point's root-mean-square distance to its $k$ nearest neighbours, so a simplex
enters according to how densely its neighbourhood is occupied rather than how
close its points are (Figure~\ref{fig:dtm-vs-rips}). This preserves more information about the local density of the point cloud, which is particularly useful in this context, as the generating parameters of SPPs directly affect local density.

As $r$ grows, homological features -- connected components ($H_0$) and loops
($H_1$) -- are \emph{born} and later \emph{die}
by merging. Each feature may be recorded as birth/death pairs $(b,d)$ corresponding to the filtration scales at which it is born and dies. The multiset of all such pairs for homology dimension $h$ is the
\emph{persistence diagram} $D_h$; the lifetime $d-b$ is often taken as a measure of a feature's robustness, so points near the diagonal are usually read as noise. 

$H_0$ records the order and scales at which clusters merge and $H_1$ the gaps and voids the clustering leaves behind. As mentioned in the introduction to this section, persistence diagrams are multisets rather than vectors, so they must be \emph{vectorized}
before being used for machine learning.

The persistence image \citep{adams2017persistence} maps to birth-persistence coordinates $(b, d-b)$, places a weighted Gaussian bump at each point, and integrates the resulting surface over a $64\times64$ pixel grid.

\begin{figure}[ht]
\centering
\includegraphics[width=0.85\linewidth]{dtm_vs_rips.pdf}
\caption{Rips vs.\ DTM$_5$ on one clustered cloud (two blobs joined by a
sparse two-point bridge), same intermediate filtration value $r$: Rips
already merges the blobs through the bridge (left, top), while DTM$_5$
delays the merge because the bridge sits in a sparse neighbourhood
(left, bottom) -- shown in the $H_0$ diagrams (right) by the highlighted
merge event sitting much further from the diagonal's origin under DTM.}
\label{fig:dtm-vs-rips}
\end{figure}

The principal difference between features based on persistence summaries and
those based on classical second-order radial statistics is what each retains. Radially averaged statistics such as $g$ and $K$ collapse a point cloud to a
real-valued curve indexed by scale $r$, and are turned into features by
evaluating over $[r_{min}, r_{max}]$ and subsampling to obtain a vector in
$\R^n$; this discards multi-scale connectivity and void structure, i.e.\
\emph{how} the density is distributed across space. A persistence diagram
retains this structure, which is directly shaped by the SPP parameters.

% #############################################################################
\section{ParamNet}
\label{sec:progress}
% #############################################################################

The work completed so far consists of the synthetic cloud simulator, the ParamNet model and its architectural ablations, comparison with alternative vectorizations, and benchmarking against existing baselines trained on the same data.

% -----------------------------------------------------------------------------
\subsection{Synthetic cloud generation}
\label{ssec:simulator}
% -----------------------------------------------------------------------------

ParamNet is trained on synthetically-generated pairs $(\pc_i, \theta_i)$
where $\theta_i$ is drawn from a \emph{design distribution} over parameter space
and $\pc_i$ is a single realization of the corresponding process, generated by
the construction of Section~\ref{ssec:spp}. The design distribution therefore
acts as a prior, and the qualitative regimes it covers determine where the
trained model is usable.

After adopting this naive approach at first, I found that sampling each parameter uniformly and independently from some interval is inadequate. This is because the regimes of interest occupy curved subsets of the parameter grid and most of the mass landed on configurations that were near-Poisson or degenerate. 

To address this, I moved to sampling in reparameterized coordinates that allow for more direct control of these regimes: the expected intensity
$\lambda = \kappa\mu$, the parent intensity $\kappa$, and the dimensionless
overlap index $\nu = r\sqrt{\kappa}$, where $r$ is the cluster scale of the
family in question ($r=\sigma$ for Thomas, $r=\sqrt{\sigma_1^2+\sigma_2^2}$ for nested Thomas, with the extra coordinate $\rho = \sigma_2/\sigma_1$). Since $\nu$ measures cluster radius against mean inter-parent spacing, $\nu \ll 1$ gives well-separated clusters and $\nu \gtrsim 1$ clusters that merge. Draws are log-uniform, since the parameters span orders of magnitude, and are constrained so that $\lambda \in [150,800]$ (persistence computation scales badly in the number of points), $\kappa \in [15,120]$ (enough clusters per window to be informative, few enough to remain resolvable), and $\nu \in [0.10,0.90]$. The upper limit on $\nu$ deliberately retains the near-Poisson regime in which classical estimators are least stable, rather than restricting the data to well-separated clusters.

% -----------------------------------------------------------------------------
\subsection{Model: ParamNet}
\label{ssec:method}
% -----------------------------------------------------------------------------

Let $\pc$ denote a point cloud, $v$ its vectorized persistence summary, and
$e \in \R^{E + 1}$ the resulting feature vector, where $E$ denotes the encoder's
embedding dimension and the $+1$ is the appended $\log N(\pc)$. ParamNet
follows the pipeline
    \begin{equation}
    \label{eq:pipeline}
    \pc \xrightarrow{\;\text{Vectorization}\;} v
    \xrightarrow{\;\text{Encoder}\;} e
    \xrightarrow{\;\text{FCNN}\;} \hat\theta.
    \end{equation}
The pipeline shape resembles the neural-estimator patterns used in
\citet{vihrs2022neural} and \citet{yip2025learning}; what I have added is the
application to a more challenging process (nested Thomas), an explicit
evaluation of the design choices at each stage, and a comparison of TDA-based
vectorizations, on which there has been no formalized discussion in this
setting.

\paragraph{Filtration.} My starting point was the canonical Rips filtration,
but it is built from inter-point distances alone: it is sensitive to the
sparsest connection between two regions and does not register how densely those
regions are occupied. Since the parameters of a point process act on local
density, DTM$_k$ seemed the richer choice, and the experiments
confirmed this -- Rips is the worst filtration arm on both processes, by a wide margin on nested Thomas (Table~\ref{tab:branches}). 

Within DTM$_k$, error grows monotonically with $k$, consistent with a finer $k$-nearest neighbourhood retaining more of the local-density signal while a coarser one averages it away. I experimented with fusing several $k$ scales to give the encoder fine and coarse summaries simultaneously; this helps on nested Thomas but not on Thomas, so the reported models use $DTM_{k=5}$ for Thomas and fused $DTM_{k\in\{5,10,15\}}$ for nested Thomas. This per-process choice is a weakness I discuss in Section~\ref{sec:future}.

\paragraph{Encoder, head, and ablations.} For each choice of filtration I compute $H_0$ and $H_1$
persistence images. On Thomas, which uses DTM$_5$ only, the model computes two diagrams per cloud, while Nested Thomas, which uses DTM$_{\{5,10,15\}}$ requires computing six diagrams (both homology dimensions for each of the three filtrations). All images are $64\times64$. Each image is passed through a small convolutional stack: three $3\times3$ blocks with channel widths $(32,64,128)$, each followed by a ReLU and $2\times2$ max-pooling with spatial dropout ($p=0.2$) between blocks. Each block is a CoordConv layer \citep{liu2018coordconv}, concatenating two channels
holding each pixel's normalized coordinates before the convolution. I added this to the architecture because persistence image's axes are not arbitrary, so a pixel's absolute location is informative.  The feature map is reduced by adaptive average pooling and projected to $E=64$. $\log N(\pc)$ is appended, and the result is passed to an FCNN head with two hidden layers of width $64$ and $32$ (ReLU, dropout $p=0.1$). 

Every experiment uses AdamW ($\text{lr}=10^{-3}$, weight decay $3\times10^{-4}$), batch size $32$, up to $500$ epochs with
early-stopping patience $50$.

When three scales are used they share one encoder and their embeddings are concatenated. I experimented with various alternative architectures; (Table~\ref{tab:branches}) reports the various ablations attempted. Notably, when multiple filtrations are used, shared encoder beats three independent ones, which is desirable due to the significant difference in the number of trainable parameters. 

In terms of which homology dimension is most valuable, $H_0$
alone is competitive while $H_1$ alone collapses to chance. Nevertheless, both $H_0$ and $H_1$ are computed per diagram because it has been seen to provide stability in certain settings, see Figure~\ref{tab:branches}. Figure~\ref{fig:vectorization_schematic} shows the four vectorizations compared on a common diagram, and none of landscapes, silhouettes or Betti curves improved on persistence images on either process. Two further arms isolate where the signal lives: training on $\log N$ alone collapses to the chance level on both
processes, while removing $\log N$ from the full model costs little on nested
Thomas ($0.192\to0.203$) and is merely noisy on Thomas. The persistence images,
not the point count, are doing the work.

\begin{figure}[tbp]
    \centering
    \includegraphics[width=\textwidth]{vectorization_schematic.pdf}
    \caption{One point cloud $\rightarrow$ its $H_0/H_1$ persistence diagrams
    $\rightarrow$ the vectorizations compared in Table~\ref{tab:branches}:
    persistence-image raster, landscape stack, silhouette curve, Betti curve,
    and persistence-statistics summary. Top row: the tight-cluster design
    ($\kappa=50$, $\mu=4$, $\sigma=0.01$); bottom row: the near-Poisson design
    ($\kappa=4$, $\mu=5$, $\sigma=0.5$). Statistics-panel bars are min-max
    normalized per statistic, with raw values printed above each bar, since
    count, total persistence, and entropy live on incomparable scales.}
    \label{fig:vectorization_schematic}
\end{figure}

\begin{table}[tbp]
\centering
\footnotesize
\begin{tabular}{llcc}
\toprule
Axis & Arm & Thomas $\mathcal{L}$ & Nested-Thomas $\mathcal{L}$ \\
\midrule
Filtration & Rips (vec\_multik, landscape) & $0.178 \pm 0.008$ & $0.461 \pm 0.016$ \\
           & DTM$_5$ / DTM$_{15}$ & $0.125 \pm 0.007$ / $0.146 \pm 0.006$ & $0.205 \pm 0.003$ / $0.260 \pm 0.017$ \\
           & DTM$_{\{5,10,15\}}$ (fused) & $0.126 \pm 0.008$ & $0.192 \pm 0.010$ \\
Homology   & $H_0$ only / $H_1$ only & $0.125 \pm 0.006$ / $0.900 \pm 0.019$ & $0.280 \pm 0.250^\dagger$ / $0.932 \pm 0.028$ \\
Encoder    & shared / independent, concat & $0.125 \pm 0.007$ / $0.134 \pm 0.009$ & $0.201 \pm 0.010$ / $0.222 \pm 0.021$ \\
Vectorization & Persistence image & $0.126 \pm 0.008$ & $0.192 \pm 0.010$ \\
           & Landscape & $0.132 \pm 0.008$ & $0.250 \pm 0.005$ \\
           & Silhouette ($p=1$) & $0.165 \pm 0.013$ & $0.271 \pm 0.012$ \\
           & Betti curve & $0.196 \pm 0.014$ & $0.290 \pm 0.005$ \\
Features   & $\log N$ removed & $0.217 \pm 0.278^\dagger$ & $0.203 \pm 0.007$ \\
           & Topology removed ($\log N$ only) & $0.893 \pm 0.015$ & $0.948 \pm 0.019$ \\
\bottomrule
\end{tabular}
\caption{Branch summary: every axis measured against the multi-scale,
$H_0{+}H_1$, CoordConv starting configuration, $n=10$ seeds per arm unless
noted. Vectorization rows are the native-encoder path. A dagger ($^\dagger$)
marks an arm on which at least one seed failed to converge and collapsed to the
chance level, so the mean is dominated by the failed seed(s). The DTM$_5$ / fused rows support the process-specific scale choice
rather than ablating a single model.}
\label{tab:branches}
\end{table}

% -----------------------------------------------------------------------------
\subsection{Estimation performance and baseline comparison}
\label{ssec:results}
% -----------------------------------------------------------------------------

I compared ParamNet against the classical and neural baselines mentioned in 
Section~\ref{sec:intro}: minimum contrast fit to Ripley's $K$ and, separately,
to the pair correlation function $g$, and \texttt{vihrs} -- my re-implementation
of the neural estimator from \citet{vihrs2022neural}: a 1-D CNN over the
edge-corrected, centred Ripley $L(r)-r$ curve plus $\log N(\pc)$, i.e.\ the
pipeline of Section~\ref{ssec:method} with the persistence-image feature
replaced by a single classical radial summary. All methods are run on the same
clouds and splits. Chance level is measured by retraining
the default model on randomly permuted labels.

\texttt{vihrs} beats ParamNet on Thomas: its per-parameter losses
($\mathcal{L}^\kappa=0.192$, $\mathcal{L}^\mu=0.122$,
$\mathcal{L}^\sigma=0.029$) are each individually lower than ParamNet's own
($0.201$, $0.134$, $0.040$), see Table~\ref{tab:perparam}). So on this single-level
process, the topological pipeline does not beat a hand-built radial summary. However, on
nested Thomas, ParamNet gains an advantage on the density and count targets
while \texttt{vihrs} maintains an edge on both scale parameters. $L(r)-r$ is
built to resolve a characteristic radius directly, while a density-based
topological feature is not, and it is only on the more complex two-level process
that this trade-off favours the topological pipeline.

Both neural methods dominate minimum contrast on Thomas by roughly an order of
magnitude, but for two different reasons. On $K$ the estimator achieves losses
of $0.365$--$0.399$ for $6/10$ seeds -- close to \texttt{vihrs} -- and
$0.903$--$2.056$ on the remaining $4/10$, which is the near-Poisson flatness
described in Section~\ref{sec:intro}. On $g$ every seed is worse than chance,
indicating systematic bias rather than scattered non-convergence; the likely
cause is my contrast exponent $c=1$, set by analogy with
\citet{diggle2003spatial}'s $K$-based rule of thumb rather than tuned, where
$c=1/4$ is the value validated for aggregated patterns. I report the number but
do not read it as a statement about contrast estimation on $g$ in general.

Finally, parent intensity $\kappa$ is the hardest target and the cluster scale
$\sigma$ the easiest, for every method on both processes and by a wide margin
(Figure~\ref{fig:paramnet-scatter}).
This ordering holds regardless of estimator, so it looks like a property of the
estimation problem itself rather than an artifact of any one feature set or
architecture.

\begin{figure}[ht]
\centering
\includegraphics[width=0.97\linewidth]{paramnet_scatter.pdf}
\caption{Predicted versus true parameter values, ParamNet (DTM$_5$,
persistence image) for the Thomas process (one seed). Both axes in
log-normalized units; identity line dashed. The difficulty of predicting parent intensity
is visually evident: we see a wider cloud for $\kappa$ compared to the tighter diagonal bands for $\mu$ and $\sigma$.}
\label{fig:paramnet-scatter}
\end{figure}

\begin{table}[tbp]
\centering
\small
\begin{tabular}{lcc}
\toprule
Method & Thomas $\mathcal{L}$ & Nested-Thomas $\mathcal{L}$ \\
\midrule
ParamNet$^{\ddagger}$    & $0.125 \pm 0.007$ & $0.192 \pm 0.010$ \\
vihrs ($L(r)-r$ CNN)     & $0.115 \pm 0.007$ & $0.217 \pm 0.006$ \\
Minimum contrast, $K$   & $0.855 \pm 0.682^{a}$ & ---$^{d}$ \\
Minimum contrast, $g$   & $6.199 \pm 0.339^{b}$ & ---$^{d}$ \\
\midrule
\emph{Chance-level reference (label shuffle)} & \emph{$1.003 \pm 0.019$} & \emph{$1.006 \pm 0.020$} \\
\bottomrule
\end{tabular}
\caption{Estimator comparison on Thomas and Nested Thomas, $n=10$ seeds, on the
shared test split. $^{\ddagger}$ParamNet is process-specific: DTM$_5$ for
Thomas, fused DTM$_{5,10,15}$ for nested Thomas. $^a$ Minimum contrast on $K$ is
unstable across seeds rather than uniformly bad, and $^b$ minimum contrast on
$g$ is consistently worse than chance; both are discussed in the text.
$^{d}$Not run: no closed-form summary statistic for nested Thomas has yet been
derived, so its absence is an outstanding experiment rather than an
impossibility.}
\label{tab:classical-comparison}
\end{table}

\begin{table}[tbp]
\centering
\small
\textbf{(a) Thomas} ($\kappa$ parent intensity, $\mu$ offspring intensity, $\sigma$ cluster scale)\\[2pt]
\begin{tabular}{lcccc}
\toprule
Parameter & ParamNet & vihrs & Min.\ contrast, $K$ & Min.\ contrast, $g$ \\
\midrule
$\mathcal{L}^\kappa$ & $0.201 \pm 0.010$ & $0.192 \pm 0.013$ & $1.373 \pm 1.087$ & $8.825 \pm 0.485$ \\
$\mathcal{L}^\mu$    & $0.134 \pm 0.007$ & $0.122 \pm 0.007$ & $0.939 \pm 0.758$ & $6.334 \pm 0.370$ \\
$\mathcal{L}^\sigma$ & $0.040 \pm 0.004$ & $0.029 \pm 0.002$ & $0.254 \pm 0.203$ & $3.437 \pm 0.170$ \\
\bottomrule
\end{tabular}

\vspace{10pt}
\textbf{(b) Nested Thomas}\\[2pt]
\begin{tabular}{lcc}
\toprule
Parameter & ParamNet$^{c}$ & vihrs \\
\midrule
parent intensity      & $0.283 \pm 0.023$ & $0.320 \pm 0.010$ \\
meta-offspring        & $0.402 \pm 0.022$ & $0.493 \pm 0.022$ \\
meta-cluster scale    & $0.143 \pm 0.006$ & $0.132 \pm 0.009$ \\
mean offspring        & $0.086 \pm 0.004$ & $0.110 \pm 0.010$ \\
cluster scale         & $0.044 \pm 0.000$ & $0.028 \pm 0.003$ \\
\bottomrule
\end{tabular}
\caption{Per-parameter loss $\mathcal{L}^j$ underlying the aggregate
$\mathcal{L}$ of Table~\ref{tab:classical-comparison} (same methods, same
seeds; ParamNet/nested is $n{=}3$). Minimum contrast has no nested-Thomas
columns for the reason given there.}
\label{tab:perparam}
\end{table}

\newpage

% #############################################################################
\section{Limitations and future work}
\label{sec:future}
% #############################################################################

The results support three claims. Persistence images are informative about
Neyman-Scott parameters: both processes sit roughly an order of magnitude below
the label-shuffle ceiling, and the two ``Features'' arms of
Table~\ref{tab:branches} place that signal in the images rather than in
$\log N$. However, they do not universally beat a well-chosen radial summary on a
single-level process: \texttt{vihrs} wins on all three Thomas parameters, and on
both scale parameters of nested Thomas. The advantage is confined to the
two-level process and to its density and count targets, which is what one would
expect of a feature built from local density rather than from a characteristic
radius. Two secondary observations: the ordering of parameter difficulty is
stable across feature sets and architectures; seemingly a property of the
estimation problem.

\paragraph{Limitations.}
\begin{itemize}[leftmargin=*,itemsep=2pt]
    \item \textbf{Multiple models.} The number and choice of filtration scales is a
    per-process choice with no established rule beyond experimentation and
    comparison of validation losses, which undercuts my process-agnostic
    objective.
    \item \textbf{Competitiveness.} Three alternative vectorizations were
    implemented and none improved on persistence images. This is evidence about
    these summaries under this architecture and this design distribution, not necessarily a
    general claim about vectorizations. More work should be done on constructing optimal architectures for each vectorization.
    \item \textbf{No advantage on Thomas.} \texttt{vihrs} wins on every
    parameter of the single-level process, and on both scale parameters of the
    nested one.
    \item \textbf{The design distribution is a prior.} The trained model
    approximates a posterior summary under it, so its ranges must cover the
    intended application domain.
    \item \textbf{Computational cost.} Persistence computation scales badly in
    $n$; this is why the point budget of Section~\ref{ssec:simulator} is capped
    at 800. Furthermore, the inference cost per cloud is currently unmeasured.
\end{itemize}

\paragraph{Planned directions.}
\begin{itemize}[leftmargin=*,itemsep=2pt]
    \item \textbf{Encoder architecture.} Aim to design a universal architecture that
    is process-agnostic.
    \item \textbf{More processes.} Inhomogeneous / non-stationary processes, where classical closed forms
    are least available and the topological approach has the most to gain. The
    comparator is the two-step procedure of \citet{waagepetersen2009twostep},
    which needs the covariates observed and the intensity correctly specified; ParamNet needs neither.
    \item \textbf{Apply to real data.} Galaxy catalogues if the running example of
    Section~\ref{sec:intro} is cosmological. The Barro Colorado Island tree data
    of \citet{waagepetersen2009twostep} might be a second target: a small public
    data set already fitted with a Thomas process.
\end{itemize}

\newpage

% #############################################################################
\part{Hessian Spectra During Grokking}
% #############################################################################

% #############################################################################
\section{Grokking and motivation}
\label{sec:hess-intro}
% #############################################################################

Grokking was first reported by \citet{power2022grokking}. It is a phenomenon by which a model attains near-perfect training
accuracy quickly ($\sim\!10^3$ optimization steps), then remains at chance-level test accuracy for an extended period ($10^4$--$10^5$ steps),
and then transitions sharply to high test accuracy long after the training loss has saturated.  This extreme delay is the object of interest: whatever separates the memorizing phase from the generalizing one is not directly visible in the training loss, which suggests that one must study the local geometry of the loss landscape to understand this sudden transition. In this project I used the Hessian's eigenvalue spectrum as
a description of this geometry, following \citet{fort2018goldilocks}, and
estimated it by the Lanczos-based approach of \citet{yao2020pyhessian}, which
recovers spectral information without explicitly forming the Hessian, which is a gigantic object for anything beyond a simple low-parameter toy model. Since this was my first project, I believed it would be instructive to implement the approximation machinery directly rather than using the PyHessian library directly. 

During my preliminary explorations, I trained a simple feed-forward network on MNIST and FashionMNIST. In that phase I observed what appeared to be a \emph{spectral collapse}, with eigenvalues above $\sim\!1$ disappearing around the grokking transition. I later moved on from this setting, as the literature -- \citet{gromov2023grokking} in particular -- treats modular arithmetic as a more interesting and appropriate context for this problem.

The current experiments use modular addition ($a + b \bmod p$) with a small
decoder-only transformer trained with AdamW and weight decay large enough to
penalize the pure-memorization solution. Training accuracy reaches
$\sim\!100\%$ within a few thousand optimization steps, while test accuracy
remains near chance ($1/p$) for between hundreds and tens of thousands of
further steps before rising above $90\%$ (Figure~\ref{fig:hess-wd-curve}). The length of that delay, and its
dependence on weight decay and initialization scale, is the phenomenon this part
of the review addresses.

\begin{figure}[ht]
\centering
\includegraphics[width=0.97\linewidth]{wd_curve_phenomenon.pdf}
\caption{Train and test accuracy vs. optimization step (log scale) for the modular-arithmetic task ($p=91$, wd$=0.5$), mean $\pm$ std over the 10 logged seeds. Training saturates almost immediately, test accuracy sits flat at chance ($1/91$) for roughly three decades of steps, then rises sharply once the network groks — establishing the qualitative shape of the phenomenon before any spectral quantity is introduced.}
\label{fig:hess-wd-curve}
\end{figure}

% #############################################################################
\section{Spectral observables}
\label{sec:hess-observables}
% #############################################################################

For network parameters $\theta \in \R^n$ and loss function $\mathcal{L}(\theta)$, the Hessian is
$\bH = \nabla^2_\theta \mathcal{L}(\theta) \in \R^{n\times n}$, symmetric with
eigenvalues $\lambda_1 \le \cdots \le \lambda_n$. Negative curvature. Rather than individual eigenvalues, eight scalar summaries of the full spectrum are tracked at a series of checkpoints:

\begin{enumerate}[leftmargin=*,itemsep=1pt,label=(\roman*)]
    \item \texttt{top\_eig}: the largest eigenvalue $\lambda_n$.
    \item \texttt{bulk\_edge}: a robust threshold separating the bulk of the
    spectrum from its outliers.
    \item \texttt{n\_outliers}: the number of eigenvalues above
    \texttt{bulk\_edge}.
    \item \texttt{trace}: $\mathrm{tr}\,\bH = \sum_i \lambda_i$.
    \item \texttt{spectral\_entropy}: the Shannon entropy of the normalised
    spectral measure.
    \item \texttt{effective\_rank}: the exponential of \texttt{spectral\_entropy}.
    \item \texttt{negative\_mass}: the spectral mass on meaningfully negative
    eigenvalues.
    \item \texttt{conditioning}: the ratio \texttt{top\_eig}/\texttt{bulk\_edge}.
\end{enumerate}

\newpage 

% #############################################################################
\section{Spectral estimation}
\label{sec:hess-lanczos}
% #############################################################################

For a network with a number of parameters in the $10^3$--$10^7$ range, forming
$\bH$ explicitly is infeasible, so the observables above are estimated with
stochastic Lanczos quadrature. Writing
$g(\theta) = \nabla_\theta \mathcal{L}(\theta)$, and given a Krylov depth $m$
and probe count $k$, the procedure is:

\begin{enumerate}[leftmargin=*,itemsep=1pt]
    \item \textbf{Hessian--vector products.} Pearlmutter's identity gives
    $\bH v = \nabla_\theta\!\left(g(\theta)^\top v\right)$ from two backward
    passes, so $\bH$ is only ever accessed through the map $v \mapsto \bH v$.
    \item \textbf{Lanczos.} For each probe $j = 1,\dots,k$: draw a random unit
    vector $v^{(j)}$ and run $m$ steps of the Lanczos recurrence, calling
    step~1 once per step, to obtain a tridiagonal
    $T^{(j)} \in \R^{m \times m}$ --- the projection of $\bH$ onto the order-$m$
    Krylov subspace generated by $v^{(j)}$.
    \item \textbf{Quadrature rule.} Eigendecompose
    $T^{(j)} = U^{(j)} \Lambda^{(j)} U^{(j)\top}$. The Ritz values
    $\lambda_i^{(j)} = \Lambda^{(j)}_{ii}$ are the quadrature nodes and
    $w_i^{(j)} = \big(U^{(j)}_{1i}\big)^2$ the weights; this rule matches the
    first $2m-1$ moments of the spectral measure of $\bH$ seen by $v^{(j)}$.
    \item \textbf{Pooling.} Average over probes,
    $\hat\mu = \frac{1}{k}\sum_{j=1}^{k}\sum_{i=1}^{m} w_i^{(j)}\,
    \delta_{\lambda_i^{(j)}}$, giving an estimate of the full spectral measure.
    \item \textbf{Observables.} Every quantity of
    Section~\ref{sec:hess-observables} is a functional of $\hat\mu$; for
    instance $\widehat{\mathrm{tr}}\,\bH = n\!\int\!\lambda \, d\hat\mu$.
\end{enumerate}

The Krylov depth $m$ and probe count $k$ are the two parameters trading compute
against accuracy.

\newpage

% #############################################################################
\section{Estimator validation}
\label{sec:hess-validation}
% #############################################################################

All eight observables were validated against two exact-reference fixtures with
known ground-truth spectra: a small MLP whose Hessian can be materialized in full. The validation identified two estimator
defects, distinct from mere under-convergence. First, the original
\texttt{bulk\_edge} and \texttt{conditioning} definitions used an unweighted
median and MAD over the Ritz values, which was biased by 30--50\% at every
Krylov depth: Lanczos node placement is moment-matching rather than
density-matching, so an unweighted median over-trusts the sparse nodes near the
spectral edge. Second, the original \texttt{negative\_mass} definition
integrated mass immediately either side of the zero crossing and therefore
measured Lanczos noise rather than curvature; it is now margin-thresholded. Both
are corrected in the current code. Table~\ref{tab:hess-mk} reports the resulting
per-observable convergence requirements.

\begin{table}[htbp]
\centering
\small
\begin{tabular}{lccc}
\toprule
Observable & Governed by & Minimum $m$ & Minimum $k$ \\
\midrule
trace & $k$ (Monte Carlo, $\propto k^{-1/2}$) & irrelevant & $\ge 100$ \\
top\_eig & $m$ & $\ge 20$ & $1$ \\
negative\_mass (margin-thresholded) & $m$ & $\ge 30$--$50$ & $\ge 30$ \\
spectral\_entropy / effective\_rank & $m$ & $\ge 20$--$30$ & $\ge 30$ \\
bulk\_edge / conditioning (post-fix) & $m$ & $\ge 75$--$100$ & not yet swept \\
\bottomrule
\end{tabular}
\caption{Approximation accuracy against Lanczos depth $m$ and probe count $k$,
measured on the exact-reference fixtures
(\texttt{reports/spectral\_validation.md}). Production defaults are $m=100$,
$k=100$, set by \texttt{bulk\_edge}/\texttt{conditioning}, the deepest-converging
pair.}
\label{tab:hess-mk}
\end{table}

This analysis is carried out entirely on synthetic fixtures of roughly $10^3$
parameters. It has not yet been validated against the spectrum of a trained
network on the modular-addition task; that check is listed among the next steps
in Section~\ref{sec:hess-next}.

% #############################################################################
\section{Experiments}
\label{sec:hess-experiments}
% #############################################################################

All current experiments run on modular addition with a small decoder-only
transformer on $a+b \bmod p$, and this setting is the source of every measured
number reported below. Two hyperparameters control the onset of grokking on this
task, and both were
swept with logged runs (AdamW, effectively full-batch, $p=91$, $150{,}000$
steps). Weight decay was swept from $0.45$ to $1.0$ in steps of $0.05$, ten
seeds per setting. Initialization scale was swept at $\alpha \in \{1,8\}$, where
every parameter is multiplied by the scalar $\alpha$ after standard
initialization. The trends are monotone. At fixed $\alpha$, larger weight decay
produces an earlier generalization onset: the mean onset step falls from
$\sim\!3150$ at $wd=0.45$ to $\sim\!1350$ at $wd=1.0$, under a
first-crosses-twice-chance onset definition. At fixed $wd=1.0$, $\alpha=8$
roughly doubles the onset delay relative to $\alpha=1$ ($\sim\!2890$ against
$\sim\!1350$ steps).

% #############################################################################
\section{Current results}
\label{sec:hess-status}
% #############################################################################

The Hessian spectrum has been measured against checkpoints from the
modular-addition sweep, at two settings: $p=91$, $wd=0.6$ (onset step 3000, 66
checkpoints) and $wd=1$ (onset step 2400, 150 checkpoints), both at the same seed.
Figure~\ref{fig:eff-rank} aligns both runs on their own onset step.
\texttt{effective\_rank} is the observable that agrees between them: it
collapses by $13.4\times$ after onset at $wd=0.6$ and by $14.2\times$ at
$wd=1$ --- two independent runs landing within 6\% of each other on the size
of the drop. \texttt{top\_eig\_corr} also rises sharply in both ($2.7\times$
and $16.1\times$), though far less consistently in magnitude.
\texttt{trace\_corr}, by contrast, disagrees in sign and timing between the
two runs and does not survive an alternate-normalizer check, so it is
omitted from the figure and not reported as a result here (see
\texttt{reports/figures/gate\_p91\_wd\{0.6,1\}\_seed0\_robustness.png} for
that check).

This is $n=1$ per condition with no error bars, and a further 5 of 66
checkpoints at $wd=0.6$ (19 of 150 at $wd=1$) have a training-accuracy dip
below 0.9 flagged as unreliable (hollow markers below); a same-$wd$,
multi-seed replication with those checkpoints held out is the natural next
check (Section~\ref{sec:hess-next}).

\begin{figure}[ht]
\centering
\begin{subfigure}[t]{0.48\linewidth}
\centering
\includegraphics[width=\linewidth]{onset_vs_wd.pdf}
\caption{Mean generalization onset step (first crossing of 50\% test accuracy) against weight decay, for initialization scale $\alpha=1$ and $\alpha=8$, ten seeds per setting shown as individual points. For $\alpha=1$, onset decreases roughly monotonically from $\sim$8.7k to $\sim$3.5k steps as weight decay rises from 0.45 to 1.0. For $\alpha=8$ (incomplete: only run at wd$\geq$0.7 for now) the decrease is present but noisier and not strictly monotone, and $\alpha=8$'s onset sits above $\alpha=1$'s by a factor of roughly 1.1--1.6$\times$ rather than a single constant offset.}
\label{fig:onset-wd}
\end{subfigure}
\hfill
\begin{subfigure}[t]{0.48\linewidth}
\centering
\includegraphics[width=\linewidth]{effective_rank_compare.pdf}
\caption{Effective rank (top) and top eigenvalue (bottom), both $\|\theta\|^{-2}$-corrected and normalized to a reference step, plotted against steps since each run's own onset (2$\times$-chance crossing: step 3000 at $wd=0.6$, step 2400 at $wd=1$), so the two runs overlay regardless of where onset falls in absolute step count. Hollow markers flag checkpoints with training accuracy below 0.9 (5/66 at $wd=0.6$, 19/150 at $wd=1$). Effective rank collapses by a similar factor in both runs after onset ($13.4\times$ at $wd=0.6$, $14.2\times$ at $wd=1$); the top eigenvalue rises in both but far less consistently in magnitude ($2.7\times$ vs.\ $16.1\times$). Both series are noisy step to step even at unflagged checkpoints, so the collapse factors should be read as order-of-magnitude agreement against a noisy background, not a clean trend. $n=1$ seed per condition, no error bars; \texttt{trace\_corr} is omitted (see text).}
\label{fig:eff-rank}
\end{subfigure}
\caption{Weight decay controls grokking onset across the accuracy-only sweep
(a); the two real spectral runs of Section~\ref{sec:hess-status} (b) sit at
$wd=0.6$ and $wd=1$ within that same sweep.}
\label{fig:onset-eff-compare}
\end{figure}

% #############################################################################
\section{Next steps}
\label{sec:hess-next}
% #############################################################################

The two gaps identified on returning to this project -- no accuracy sweep for
the $(m,k)$ approximation, no weight-norm normalization despite the
$\|\theta\|^{-2}$ scaling -- are both closed at the level of tested code and
applied to the two real runs of Section~\ref{sec:hess-status}.

\paragraph{Limitations.}
\begin{itemize}[leftmargin=*,itemsep=2pt]
    \item \textbf{Single seed per condition.} Both real runs are $n=1$; the
    \texttt{effective\_rank} agreement of Section~\ref{sec:hess-status} could
    be a coincidence of two seeds rather than a stable effect.
    \item \textbf{Estimator accuracy untested on this task.} The $(m,k)$
    requirements of Table~\ref{tab:hess-mk} come from synthetic fixtures
    (Section~\ref{sec:hess-validation}), not from the modular-addition network
    itself.
    \item \textbf{Trace inconclusive.} \texttt{trace\_corr} disagrees in sign
    and timing between the two runs and fails the alternate-normalizer check;
    it is not currently a usable observable.
    \item \textbf{Coverage.} $9{,}465$ checkpoints from the weight-decay sweep
    are already on disk, and only two $(wd,\text{seed})$ settings have been
    run through \texttt{scripts/spectral\_from\_checkpoints.py} so far.
\end{itemize}

\paragraph{Planned directions.}
\begin{itemize}[leftmargin=*,itemsep=2pt]
    \item \textbf{Replicate.} More seeds at $wd\in\{0.6,1\}$, the direct test
    of whether the \texttt{effective\_rank} agreement holds at $n>1$ or was a
    coincidence of two runs.
    \item \textbf{Scale up.} The rest of the $wd$ sweep, to replace today's
    single-seed points with error bars and test the recalled collapse of
    Section~\ref{sec:hess-intro} more broadly.
    \item \textbf{Generalization bound.} \citet{tuci2026generalization}
    derives a generalization bound from the full Hessian spectrum and
    connects it explicitly to grokking, the closest existing link to the
    question motivating this work.
\end{itemize}

% #############################################################################
% REFERENCES (single unified bibliography, natbib author--year)
% #############################################################################
{\footnotesize
\setlength{\bibsep}{1pt plus 0.3ex}
\begin{thebibliography}{9}
\setlength{\itemsep}{0pt}
\setlength{\parskip}{0pt}
\bibitem[Adams et al.(2017)]{adams2017persistence}
H.~Adams et al.
\newblock Persistence images: a stable vector representation of persistent
homology.
\newblock \emph{Journal of Machine Learning Research}, 18(8):1--35, 2017.

\bibitem[Anai et al.(2020)]{anai2020dtm}
H.~Anai, F.~Chazal, M.~Glisse, Y.~Ike, H.~Inakoshi, R.~Tinarrage, and
Y.~Umeda.
\newblock DTM-based filtrations.
\newblock In \emph{Topological Data Analysis}, Abel Symposia 15, pages 33--66.
Springer, 2020.

\bibitem[Bubenik(2015)]{bubenik2015landscapes}
P.~Bubenik.
\newblock Statistical topological data analysis using persistence landscapes.
\newblock \emph{Journal of Machine Learning Research}, 16:77--102, 2015.

\bibitem[Chazal et al.(2014)]{chazal2014silhouette}
F.~Chazal, B.~T. Fasy, F.~Lecci, A.~Rinaldo, and L.~Wasserman.
\newblock Stochastic convergence of persistence landscapes and silhouettes.
\newblock In \emph{Proc.\ 30th Annual Symposium on Computational Geometry},
pages 474--483, 2014.

\bibitem[Diggle(2003)]{diggle2003spatial}
P.~J. Diggle.
\newblock \emph{Statistical Analysis of Spatial Point Patterns}.
\newblock Arnold, London, 2nd edition, 2003.

\bibitem[Diggle et al.(2005)]{diggle2005surveillance}
P.~J. Diggle, B.~Rowlingson, and T.-L. Su.
\newblock Point process methodology for on-line spatio-temporal disease
surveillance.
\newblock \emph{Environmetrics}, 16(5):423--434, 2005.

\bibitem[Fort and Scherlis(2018)]{fort2018goldilocks}
S.~Fort and A.~Scherlis.
\newblock The Goldilocks zone: towards better understanding of neural network
loss landscapes.
\newblock arXiv:1807.02581, 2018. Published at AAAI-19.

\bibitem[Gromov(2023)]{gromov2023grokking}
A.~Gromov.
\newblock Grokking modular arithmetic.
\newblock arXiv:2301.02679, 2023.

\bibitem[Liu et al.(2018)]{liu2018coordconv}
R.~Liu et al.
\newblock An intriguing failing of convolutional neural networks and the
CoordConv solution.
\newblock In \emph{Advances in Neural Information Processing Systems}, 2018.

\bibitem[Liu et al.(2023)]{liu2023regularizing}
Y.~Liu, S.~Yu, and T.~Lin.
\newblock Regularizing deep neural networks with stochastic estimators of
Hessian trace.
\newblock arXiv:2208.05924, 2022. Published in \emph{Neurocomputing},
536:13--20, 2023.

\bibitem[M\o ller and Waagepetersen(2004)]{moller2004statistical}
J.~M\o ller and R.~P. Waagepetersen.
\newblock \emph{Statistical Inference and Simulation for Spatial Point
Processes}.
\newblock Chapman \& Hall/CRC, 2004.

\bibitem[Neyman and Scott(1952)]{neyman1952theory}
J.~Neyman and E.~L. Scott.
\newblock A theory of the spatial distribution of galaxies.
\newblock \emph{Astrophysical Journal}, 116:144--163, 1952.

\bibitem[Owen et al.(2010)]{owen2010palm}
D.~M. Owen, C.~Rentero, J.~Rossy, A.~Magenau, D.~Williamson, M.~Rodriguez, and
K.~Gaus.
\newblock PALM imaging and cluster analysis of protein heterogeneity at the cell
surface.
\newblock \emph{Journal of Biophotonics}, 3(7):446--454, 2010.

\bibitem[Power et al.(2022)]{power2022grokking}
A.~Power, Y.~Burda, H.~Edwards, I.~Babuschkin, and V.~Misra.
\newblock Grokking: generalization beyond overfitting on small algorithmic
datasets.
\newblock arXiv:2201.02177, 2022.

\bibitem[Reininghaus et al.(2015)]{reininghaus2015kernel}
J.~Reininghaus, S.~Huber, U.~Bauer, and R.~Kwitt.
\newblock A stable multi-scale kernel for topological machine learning.
\newblock In \emph{IEEE Conference on Computer Vision and Pattern Recognition},
pages 4741--4748, 2015.

\bibitem[Tanaka et al.(2008)]{tanaka2008parameter}
U.~Tanaka, Y.~Ogata, and D.~Stoyan.
\newblock Parameter estimation and model selection for Neyman-Scott point
processes.
\newblock \emph{Biometrical Journal}, 50(1):43--57, 2008.

\bibitem[Tuci et al.(2026)]{tuci2026generalization}
M.~Tuci, C.~Korkmaz, U.~\c{S}im\c{s}ekli, and T.~Birdal.
\newblock Generalization at the edge of stability.
\newblock arXiv:2604.19740, 2026.

\bibitem[Umeda(2017)]{umeda2017time}
Y.~Umeda.
\newblock Time series classification via topological data analysis.
\newblock \emph{Transactions of the Japanese Society for Artificial
Intelligence}, 32(3):1--12, 2017.

\bibitem[Vihrs(2022)]{vihrs2022neural}
N.~Vihrs.
\newblock Using neural networks to estimate parameters in spatial point process
models.
\newblock arXiv preprint arXiv:2109.15056, 2022.

\bibitem[Waagepetersen and Guan(2009)]{waagepetersen2009twostep}
R.~Waagepetersen and Y.~Guan.
\newblock Two-step estimation for inhomogeneous spatial point processes.
\newblock \emph{Journal of the Royal Statistical Society: Series B},
71(3):685--702, 2009.

\bibitem[Yao et al.(2020)]{yao2020pyhessian}
Z.~Yao, A.~Gholami, K.~Keutzer, and M.~Mahoney.
\newblock PyHessian: neural networks through the lens of the Hessian.
\newblock arXiv:1912.07145, 2020.

\bibitem[Yip et al.(2025)]{yip2025learning}
J.~H.~T. Yip, A.~Rouhiainen, and G.~Shiu.
\newblock Learning from topology: cosmological parameter estimation from the
large-scale structure.
\newblock \emph{Machine Learning: Science and Technology}, 6:025063, 2025.
\end{thebibliography}
}

\end{document}