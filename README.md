# PSDMatrices.jl
A Julia package for positive semi-definite matrices.

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/PSDMatrices.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/PSDMatrices.jl/dev)
[![Build Status](https://github.com/nathanaelbosch/PSDMatrices.jl/workflows/CI/badge.svg)](https://github.com/nathanaelbosch/PSDMatrices.jl/actions)
[![Coverage](https://codecov.io/gh/nathanaelbosch/PSDMatrices.jl/branch/main/graph/badge.svg?token=PVYADY2WAX)](https://codecov.io/gh/nathanaelbosch/PSDMatrices.jl)
<!-- [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) -->
<!-- [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) -->


Positive semi-definite matrices are widely used to describe [covariance matrices](https://en.wikipedia.org/wiki/Covariance_matrix) in probabilistic models.
For strictly positive definite matrices, [PDMats.jl](https://github.com/JuliaStats/PDMasemi-ts.jl) provides a powerful Julia interface, but in many algorithms and applications, the covariances are not necessarily of full rank.
/PSDMatrices.jl/ aims to fill this gap by providing a datatype for positive semi-definite matrices.

Another major difference between /PDMats.jl/ and /PSDMatrices.jl/ is that /PSDMatrices.jl/ never assembles the full matrix $M$, but only acts on the square root matrices $M = R^*R$.

## Installation
```julia
] add PSDMatrices
```
