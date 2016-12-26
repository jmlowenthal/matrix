// Copyright (c) 2016 James Lowenthal - all rights reserved

#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#include <exception>
#include <array>
#include <cmath>

typedef unsigned int uint;

template<uint n, uint m, typename T = float> class ElementMatrix;

/**
	@brief		n x m matrix
	@param n	number of rows
	@param m	number of columns
	@param T	data type of values
	A template class for representing mathematical matricies.
*/
template<uint n, uint m, typename T = float>
class Matrix {
private:
	/**
		@brief		matrix internal data
		Internal array for holding matrix data values.
	*/
	T data[n][m];

public:
	/**
		@brief		basic matrix constructor
		Returns a zero-initialised matrix.
	*/
	Matrix();

	/**
		@brief		initialiser list constructor
		@param l	initialiser list
		Produces a matrix of correct dimensions given `n x m` items in the
		list.
	*/
	Matrix(std::array<T, n * m> l);

	/**
		@brief		matrix accessor
		@param i	row
		@param j	column
		@return		reference to value within matrix array
		Accessor for getting a location in the matrix by reference. Index
		starts from one (rather than zero) as is convention in matrices. Second
		parameter default allows easier access to single-column matricies
		(vectors).
	*/
	T& at(uint i, uint j = 1);

	/**
		@brief		matrix accessor
		@param i	row
		@param j	column
		@return		constant value from matrix array
		Accessor for getting a location in the constant matrix. Index
		starts from one (rather than zero) as is convention in matrices. Second
		parameter default allows easier access to single-column matricies
		(vectors).
	*/
	const T& at(uint i, uint j = 1) const;

	/**
		@brief		matrix accessor
		@param i	row (zero-indexed)
		@param j	column (zero-indexed)
		@return		reference to value within matrix array
		Accessor for getting a location in the matrix by reference. Index
		starts from zero, by programming convention. Second parameter default
		allows easier access to single-column matricies (vectors).
	*/
	T& at_(uint i, uint j = 0);

	/**
		@brief		matrix accessor
		@param i	row (zero-indexed)
		@param j	column (zero-indexed)
		@return		constant value within matrix array
		Accessor for getting a location in the constant matrix. Index
		starts from zero, by programming convention. Second parameter default
		allows easier access to single-column matricies (vectors).
	*/
	const T& at_(uint i, uint j = 0) const;

	/**
		@brief		matrix transpose
		@return		transposed matrix
		Transpose rows with columns of matrix.
	*/
	Matrix<m, n, T> transpose() const;

	/**
		@brief		produce a resized matrix
		@return		a matrix of specified size
		Produce of a matrix of given size with values copied from their
		respective positions in the original matrix.
	*/
	template<uint a, uint b> Matrix<a, b, T> resize() const;

	/**
		@brief		produce negated matrix
	*/
	Matrix<n, m, T> operator-() const;

	/**
		@brief		matrix multiplication
		@param rhs	matrix to multiply by
		@return		resulting matrix
	*/
	template<uint k> Matrix<n, k, T> operator*(Matrix<m, k, T>& rhs) const;

	/**
		@brief		matrix addition
		@param rhs	matrix to add
		Adds each element of this matrix to the element in the corresponding
		location in the `rhs` matrix.
	*/
	Matrix<n, m, T> operator+(const Matrix<n, m, T>& rhs) const;
	Matrix<n, m, T>& operator+=(const Matrix<n, m, T>& rhs);

	/**
		@brief		matrix subtraction
		@param rhs	matrix to subtract
		Subtracts each element of the `rhs` element from the corresponding
		location in this matrix.
	*/
	Matrix<n, m, T> operator-(const Matrix<n, m, T>& rhs) const;
	Matrix<n, m, T>& operator-=(const Matrix<n, m, T>& rhs);

	/**
		@brief		matrix scaling
		@param rhs	scaling parameter
		Multiply each element of this matrix by the scaling parameter.
	*/
	Matrix<n, m, T> operator*(const T& rhs) const;
	Matrix<n, m, T>& operator*=(const T& rhs);

	/**
		@brief		matrix scaling
		@param rhs	scaling parameter
		Divide each element of this matrix by the scaling parameter.
	*/
	Matrix<n, m, T> operator/(const T& rhs) const;
	Matrix<n, m, T>& operator/=(const T& rhs);

	/**
		@brief		cross product
		@param rhs	right-hand matrix
		Produces the cross product of two 3D vectors.
	*/
	Matrix<3, 1, T> operator%(Matrix<3, 1, T> rhs) const;

	/**
		@brief		sum of all elements
		Produces sum of all matrix elements.
	*/
	T sum() const;

	/**
		@brief		sum of all elements squared
		Sums all matrix elements squared.
	*/
	T magnitudeSq() const;

	/**
		@brief		matrix magnitude
		Square root of `sumSq()`.
	*/
	T magnitude() const;

	/**
		@brief		normalise matrix
		Produce a matrix with the same relative element values but with 
		`sumSq() = 1`.
	*/
	Matrix<n, m, T> normalise() const;

	/**
		@brief		convert to element matrix
		Converts matrix to element-matrix for element-wise operations.
	*/
	ElementMatrix<n, m, T> operator()() const;

	/**
	@brief		dot product
	Element-wise multiplication, followed by summation of all elements.
	*/
	T dot(Matrix<n, m, T> rhs) const;

	/**
		@brief		identity matrix
		@return		identity matrix for template dimensions
		Produces the identity matrix. This function is only defined for square
		matricies.
	*/
	static Matrix<n, m, T> identity();
};

template<uint n, uint m, typename T>
class ElementMatrix {
	friend class Matrix<n, m, T>;

private:
	/**
		@brief		matrix data source
		The matrix to which this element matrix uses a read-only source of
		data.
	*/
	const Matrix<n, m, T> & lhs;
	
	/**
		@brief		construct element matrix
		@param m	matrix data source
	*/
	ElementMatrix(const Matrix<n, m, T>& mat) : lhs(mat) {};

public:
	/**
		@brief		element-wise addition
		@param rhs	right-hand side elements
	*/
	Matrix<n, m, T> operator+(Matrix<n, m, T> rhs);
	
	/**
		@brief		element-wise addition
		@param rhs	right-hand side elements
	*/
	Matrix<n, m, T> operator-(Matrix<n, m, T> rhs);
	
	/**
		@brief		element-wise addition
		@param rhs	right-hand side elements
	*/
	Matrix<n, m, T> operator*(Matrix<n, m, T> rhs);
	
	/**
		@brief		element-wise addition
		@param rhs	right-hand side elements
	*/
	Matrix<n, m, T> operator/(Matrix<n, m, T> rhs);
};

template<unsigned int n, typename T = double>
using Vector = Matrix<n, 1, T>;

#pragma region Function definitions

template<uint n, uint m, typename T>
inline Matrix<n, m, T>::Matrix() {
	for (int i = 0; i < n * m; ++i) {
		//at_(i) = 0;
		new(&at_(0, i)) T();
	}
	/*for (uint i = 0; i < n; ++i) {
		for (uint j = 0; j < m; ++j) {
			at_(i, j) = 0;
		}
	}*/
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T>::Matrix(std::array<T, n * m> l) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			at_(i, j) = l[i * m + j];
		}
	}
}

template<uint n, uint m, typename T>
inline T& Matrix<n, m, T>::at(uint i, uint j) {
	return data[--i][--j];
}

template<uint n, uint m, typename T>
inline const T& Matrix<n, m, T>::at(uint i, uint j) const {
	return data[--i][--j];
}

template<uint n, uint m, typename T>
inline T& Matrix<n, m, T>::at_(uint i, uint j) {
	return data[i][j];
}

template<uint n, uint m, typename T>
inline const T& Matrix<n, m, T>::at_(uint i, uint j) const {
	return data[i][j];
}

template<uint n, uint m, typename T>
inline Matrix<m, n, T> Matrix<n, m, T>::transpose() const {
	Matrix<m, n, T> ret;
	for (uint i = 0; i < n; ++i) {
		for (uint j = 0; j < m; ++j) {
			ret.at_(j, i) = at_(i, j);
		}
	}
	return ret;
}

template<uint n, uint m, typename T>
template<uint a, uint b>
inline Matrix<a, b, T> Matrix<n, m, T>::resize() const {
	Matrix<a, b, T> ret;
	for (int i = 0; i < n * m; ++i) {
		ret.at_(0, i) = at_(0, i);
	}
	/*for (uint i = 0; i < n && i < a; ++i) {
		for (uint j = 0; j < m && j < b; ++j) {
			ret.at_(i, j) = at_(i, j);
		}
	}*/
	return ret;
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T> Matrix<n, m, T>::operator-() const {
	Matrix<n, m, T> ret;
	for (int i = 0; i < n * m; ++i) {
		ret.at_(0, i) = -at_(0, i);
	}
	/*for (uint i = 0; i < n; ++i) {
		for (uint j = 0; j < m; ++j) {
			ret.at_(i, j) = -at_(i, j);
		}
	}*/
	return ret;
}

template<uint n, uint m, typename T>
template<uint k>
inline Matrix<n, k, T> Matrix<n, m, T>::operator*(Matrix<m, k, T>& rhs) const {
	Matrix<n, k, T> ret;
	for (uint i = 0; i < n; ++i) {
		for (uint j = 0; j < k; ++j) {
			for (uint l = 0; l < m; ++l) {
				ret.at_(i, j) += at_(i, l) * rhs.at_(l, j);
			}
		}
	}
	return ret;
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T> Matrix<n, m, T>::operator+(const Matrix<n, m, T>& rhs) const {
	Matrix<n, m, T> A;
	for (int i = 0; i < n * m; ++i) {
		A.at_(0, i) = at_(0, i) + rhs.at_(0, i);
	}
	/*for (uint i = 0; i < n; ++i) {
		for (uint j = 0; j < m; ++j) {
			A.at_(i, j) = at_(i, j) + rhs.at_(i, j);
		}
	}*/
	return A;
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T>& Matrix<n, m, T>::operator+=(const Matrix<n, m, T>& rhs) {
	*this = *this + rhs;
	return *this;
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T> Matrix<n, m, T>::operator-(const Matrix<n, m, T>& rhs) const {
	/*Matrix<n, m, T> A;
	for (int i = 0; i < n * m; ++i) {
		A.at_(0, i) = at_(0, i) - rhs.at_(0, i);
	}*/
	/*for (uint i = 0; i < n; ++i) {
		for (uint j = 0; j < m; ++j) {
			A.at_(i, j) = at_(i, j) - rhs.at_(i, j);
		}
	}*/
	return *this + (-rhs);
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T>& Matrix<n, m, T>::operator-=(const Matrix<n, m, T>& rhs) {
	*this = *this - rhs;
	return *this;
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T> Matrix<n, m, T>::operator*(const T& rhs) const {
	Matrix<n, m, T> A;
	for (int i = 0; i < n * m; ++i) {
		A.at_(0, i) = at_(0, i) * rhs;
	}
	/*for (uint i = 0; i < n; ++i) {
		for (uint j = 0; j < m; ++j) {
			A.at_(i, j) = at_(i, j) * rhs;
		}
	}*/
	return A;
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T>& Matrix<n, m, T>::operator*=(const T& rhs) {
	*this = *this * rhs;
	return *this;
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T> Matrix<n, m, T>::operator/(const T& rhs) const {
	/*Matrix<n, m, T> A;
	for (int i = 0; i < n * m; ++i) {
		A.at_(0, i) = at_(0, i) / rhs;
	}*/
	/*for (uint i = 0; i < n; ++i) {
		for (uint j = 0; j < m; ++j) {
			A.at_(i, j) = at_(i, j) / rhs;
		}
	}*/
	return *this * (1 / rhs);
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T>& Matrix<n, m, T>::operator/=(const T& rhs) {
	*this = *this / rhs;
	return *this;
}

template<uint n, uint m, typename T>
Matrix<3, 1, T> Matrix<n, m, T>::operator%(Matrix<3, 1, T> rhs) const {
	static_assert(n == 3 && m == 1, "Cross product is only defined for R^3 vectors");
	
	Matrix<3, 1, T> A;
	A.at_(0) = at_(1) * rhs.at_(2) - at_(2) * rhs.at_(1);
	A.at_(1) = at_(2) * rhs.at_(0) - at_(0) * rhs.at_(2);
	A.at_(2) = at_(0) * rhs.at_(1) - at_(1) * rhs.at_(0);

	return A;
}

template<uint n, uint m, typename T>
inline T Matrix<n, m, T>::sum() const {
	T ans = 0;
	for (int i = 0; i < n * m; ++i) {
		ans += at_(0, i);
	}
	return ans;
}

template<uint n, uint m, typename T>
T Matrix<n, m, T>::magnitudeSq() const {
	T ans = 0;
	for (int i = 0; i < n * m; ++i) {
		ans += at_(0, i) * at_(0, i);
	}
	return ans;
}

template<uint n, uint m, typename T>
T Matrix<n, m, T>::magnitude() const {
	return std::sqrt(magnitudeSq());
}

template<uint n, uint m, typename T>
Matrix<n, m, T> Matrix<n, m, T>::normalise() const {
	return Matrix<n, m, T>(*this) / magnitude();
}

template<uint n, uint m, typename T>
ElementMatrix<n, m, T> Matrix<n, m, T>::operator()() const {
	return ElementMatrix<n, m, T>(*this);
}

template<uint n, uint m, typename T>
T Matrix<n, m, T>::dot(Matrix<n, m, T> rhs) const {
	return ((*this)() * rhs).sum();
}

template<uint n, uint m, typename T>
inline Matrix<n, m, T> Matrix<n, m, T>::identity() {
	static_assert(n == m, "Identity matrix is only defined for a square matrices");
	static Matrix<n, m, T> I;
	if (I.at_(0, 0) == 0) {
		for (uint i = 0; i < n; ++i) {
			I.at_(i, i) = 1;
		}
	}
	return I;
}

template<uint n, uint m, typename T>
Matrix<n, m, T> ElementMatrix<n, m, T>::operator+(Matrix<n, m, T> rhs) {
	Matrix<n, m, T> A;
	for (int i = 0; i < n * m; ++i) {
		A.at_(0, i) = lhs.at_(0, i) + rhs.at_(0, i);
	}
	return A;
}

template<uint n, uint m, typename T>
Matrix<n, m, T> ElementMatrix<n, m, T>::operator-(Matrix<n, m, T> rhs) {
	return *this + (-rhs);
}

template<uint n, uint m, typename T>
Matrix<n, m, T> ElementMatrix<n, m, T>::operator*(Matrix<n, m, T> rhs) {
	Matrix<n, m, T> A;
	for (int i = 0; i < n * m; ++i) {
		A.at_(0, i) = lhs.at_(0, i) * rhs.at_(0, i);
	}
	return A;
}

template<uint n, uint m, typename T>
Matrix<n, m, T> ElementMatrix<n, m, T>::operator/(Matrix<n, m, T> rhs) {
	Matrix<n, m, T> A;
	for (int i = 0; i < n * m; ++i) {
		A.at_(0, i) = lhs.at_(0, i) / rhs.at_(0, i);
	}
	return A;
}

#pragma endregion

#endif