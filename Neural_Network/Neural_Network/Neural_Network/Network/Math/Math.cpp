#include "Math.h"

namespace ML
{
	const Math::Func sigmoid = [](double x) -> double { return 1.0 / (1.0 + exp(-x)); };

	const Math::FuncPair Math::sigmoidFunc
	{ 
		sigmoid, 

		[](double x) -> double 
		{ 
			return sigmoid(x) * (1.0 - sigmoid(x)); 
		} 
	};

	const Math::FuncPair Math::softPlusFunc
	{ 
		[](double x) -> double 
		{ 
			return log(1.0f + exp(x)); 
		}, 

		[](double x) -> double 
		{ 
			return 1.0f / (1.0f + exp(-x));
		} 
	};

}
