#pragma once 

#include <cmath>
#include <functional>

namespace ML
{
    namespace Math
    {
        using Func = std::function<double(double)>;
        typedef struct
        {
            Func ACT;
            Func DERIV;

        }FuncPair;

        extern const FuncPair reluFunc;
        extern const FuncPair tanhFunc;
        extern const FuncPair sigmoidFunc;
        extern const FuncPair softPlusFunc;




    }

}
