#ifndef MATRIX_H
#define MATRIX_H

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>

namespace math
{
	using Integral_t = double;
	using Int_t = std::uint_fast32_t;

	template <Int_t rows, Int_t cols>
	class Mat;

	template <Int_t rows>
	class Vec;

	template <typename Storage, Int_t rows>
	struct Get_info
	{
		static std::array<Storage, rows>& gen_arr(const std::initializer_list<Integral_t>& list)
		{
			assert(list.size() == (Storage::rows * rows) && " The size of the initializer list must match the length of the matrix\n");

			Int_t i = 0;
			for (auto& element_i : static_arr)
			{
				for (auto& element_j : element_i)
				{
					element_j = *(list.begin() + i);
					++i;
				}
			}

			return static_arr;
		}

		static constexpr Int_t cols = Storage::rows;
		static std::array<Storage, rows> static_arr;
	};

	template <typename Storage, Int_t rows>
	std::array<Storage, rows> Get_info<Storage, rows>::static_arr{};

	template <Int_t rows>
	struct Get_info <Integral_t, rows>
	{
		static std::array<Integral_t, rows>& gen_arr(const std::initializer_list<Integral_t>& list)
		{
			assert(list.size() == rows && " The size of the initializer list must match the length of the vector\n");

			Int_t i = 0;
			for (auto& element_i : static_arr)
			{
				element_i = *(list.begin() + i);
				++i;
			}

			return static_arr;
		}

		constexpr static Int_t cols = 1;
		static std::array<Integral_t, rows> static_arr;
	};

	template <Int_t rows>
	std::array<Integral_t, rows> Get_info<Integral_t, rows>::static_arr{};

	template <Int_t row, typename Storage_t>
	class LinearInterface
	{
	public:
		virtual ~LinearInterface() = default;

		LinearInterface operator-() const
		{
			LinearInterface ret_val;

			Int_t i = 0;
			for (auto& element : ret_val)
				element = -(*this)[i++];

			return ret_val;
		}

		Storage_t& operator[] (Int_t index)
		{
			return m_arr[check_range(*this, index)];
		};

		const Storage_t& operator[] (Int_t index) const
		{
			return m_arr[check_range(*this, index)];
		}

		Storage_t* begin()
		{
			return &(m_arr[0]);
		}

		Storage_t* end()
		{
			return begin() + row;
		}

		const Storage_t* begin() const
		{
			return &(m_arr[0]);
		}

		const Storage_t* end() const
		{
			return begin() + row;
		}

		static constexpr Int_t rows = row;
		static constexpr Int_t cols = Get_info <Storage_t, row>::cols;

		static Int_t check_range(const LinearInterface& check, Int_t index)
		{
			assert((index < row) && "The given index was invalid...\n");
			return index;
		}

	protected:
		LinearInterface() = default;
		LinearInterface(const std::array<Storage_t, row>& arr) :m_arr(arr)
		{}

	public:
		std::array <Storage_t, row> m_arr{};
	};

	template <Int_t row>
	class Vec : public LinearInterface <row, Integral_t>
	{
	public:
		static constexpr Int_t rows = row, cols = 1;

		Vec(Integral_t num)
			:LinearInterface<rows, Integral_t>
			(
				[&num]()->std::array<Integral_t, rows>&
		{
			static std::array<Integral_t, rows> arr;
			for (auto& element : arr)
				element = num;

			return arr;
		}()
			)
		{}

		Vec(const std::initializer_list<Integral_t>& list)
			:LinearInterface<rows, Integral_t>(Get_info<Integral_t, rows>::gen_arr(list))
		{}

		Vec(const Vec&) = default;

		Vec()
			:Vec(0)
		{
		}

		Vec& operator=(const Vec&) = default;

		double length()
		{
			double dist = 0.0;

			for (Int_t i = 0; i < rows; ++i)
				dist += std::pow((*this)[i], 2);

			return std::pow(dist, 0.5);
		}
	};

	template <Int_t row, Int_t col>
	class Mat : public LinearInterface <row, Vec<col>>
	{
	public:
		using Storage_t = Vec<col>;
		static constexpr Int_t rows = row, cols = Get_info <Storage_t, row>::cols;

		Mat(const std::initializer_list <Integral_t>& list)
			:LinearInterface<rows, Storage_t>(Get_info<Storage_t, rows>::gen_arr(list))
		{}

		Mat(const Mat&) = default;

		Mat() = default;
	};

	// the sum of the mutlitples is smallest ( you could say there's a turning point at ) f(x) = sub/x +  x, when
	// x is square root of sub. That maens that if I were to take the multiples of let's say 100, and use 50 and 2,
	// their sum would definitely be bigger than the sum of 10 and 10, which of course is 20. The sum of 50 and 2 is 
	// of course 52, and half of that is 26. That being said, any multiples which are not 10, will have a sum that is 
	// more than 20, ans so halving anything that is more than 20 will give something that is more thatn 10.

	// It is as simple as the sum of the postitive multiples of any numbers can never be larger than the sum
	// of the number's square roots

	// I think it's times the sum of the multiples by HALF because we are talking about the sum of two numbers
	// so we couldn't times the number by something like 1/3 because that would be for 3 numbers.
	// Pretty much, half  is always larger than the sum of the roots.

	
}

template <math::Int_t rows>
double dot(const math::Vec <rows>& vec, const math::Vec <rows>& vec2)
{
	double dot_prod = 0.0;

	for (math::Int_t i = 0; i < rows; ++i)
		dot_prod += (vec[i] * vec2[i]);

	return dot_prod;
}

template <math::Int_t row_col>
math::Mat <row_col, row_col> identity()
{			
	math::Mat <row_col, row_col> ret_val;

	for (math::Int_t i = 0; i < row_col; ++i)
		ret_val[i][i] = 1.0f;

	return ret_val;
}

template <math::Int_t rows>
math::Vec<rows> operator- (const math::Vec<rows>& vec, const math::Vec<rows>& vec2)
{
	return vec + (-vec2);
}

template <math::Int_t rows>
math::Vec<rows> operator+ (const math::Vec<rows>& vec, const math::Vec<rows>& vec2)
{
	math::Vec<rows> ret_val;

	math::Int_t i = 0;
	for (auto& element : ret_val)
	{
		element = (vec[i] + vec2[i]);
		++i;
	}

	return ret_val;
}

template <math::Int_t row, math::Int_t col, math::Int_t row2>
math::Mat<col, row2> operator* (const math::Mat<row, col>& mat, const math::Mat<col, row2>& mat2)
{
	math::Mat<col, row2> ret_val;

	// dot product of the row, and column vector
	const auto& multiply_row_col	
	{

		[&mat, &mat2](math::Int_t I, math::Int_t J) -> math::Integral_t
		{
			math::Vec<mat.cols> row_vec = mat[I];
			math::Vec<mat2.rows> col_vec;

			for (math::Int_t i = 0; i < mat2.rows; ++i)
				col_vec[i] = mat2[i][J];
			
			return dot(row_vec, col_vec);

		}

	};

	// assigning that dot product to each element of the matrix / vector
	for (math::Int_t i = 0; i < ret_val.rows; ++i)
		for (math::Int_t j = 0; j < ret_val.cols; ++j)
			ret_val[i][j] = multiply_row_col(i, j);

	return ret_val;
}

template <math::Int_t row, math::Int_t col, math::Int_t row2>
math::Vec<row2> operator* (const math::Mat<row, col>& mat, const math::Vec<row2>& vec)
{
	math::Vec<row2> ret_val;

	math::Int_t i = 0;
	for (const auto& r : mat)
		ret_val[i++] = dot(r, vec);

	return ret_val;
}

template <math::Int_t rows, typename Storage>
std::ostream& operator<< (std::ostream& out, const math::LinearInterface<rows, Storage>& l)
{
	if (l.cols == 1)
		out << "{";

	char end_ln = (l.cols == 1) ? ' ' : '\n';

	for (const auto& element : l)
	{
		out << element;
		if (&(element) + 1 != l.end())
			out << ", " << end_ln;	
	}

	if (l.cols == 1)
		out << "}";

	return out;
}



// make some iterator class thingie
// make the faster square root function


#endif


// become something like this :

/*
	int cols = std::is_same_v<float, float> ? 1 : float::rows;
*/

// (the above code is assuming the case when Storage_t is the type float)

// Even though they are the same type, and ideally 1 should just be assigned to
// cols normally, the compiler is super extra, and doesn't like the look of
// it

// That being said, you should try to do things that make sense for one type
// within the block of code, and then if you want something to be specifically
// different for a float for example, then you should use template specialization.