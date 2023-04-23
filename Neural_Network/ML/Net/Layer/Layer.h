#pragma once 
#include "Net/Neurone/Neurone.h"

namespace ML
{
    enum class InitMethod
    {
        Random_Uniform = 0,
        Xavier_Normal
    };

    class Layer
    {
    public:
        Layer() = default;

        void init(uInt currentLayerIndex, uInt numLayers, uInt size, Math::FuncPair, InitMethod init); 

        uInt getLen() const;

        Bias& getBias();
        const Bias& getBias() const;

        const Layer* getFront() const;
        const Layer* getBack() const;

        Layer* getFront();
        Layer* getBack();


        void calcDeltaWeights(); 
        void applyDeltaWeights(mFloat, mFloat, uInt noDataValues);

        mFloat actFunc(mFloat) const;
        mFloat derivFunc(mFloat) const;

        void propagateOutputs(); 
        mFloat addOutputs(uInt index);

        DECLARE_ITERATORS(Neurone, Layer);

        Math::FuncPair m_func;
    
    private:
        Vec<Neurone> m_neurones;
        Var<Bias> m_bias;

        Layer* m_front;
        Layer* m_back;

    };

}

inline std::ostream& operator<<(std::ostream&, const ML::Layer& l)
{
    for (const auto& n : l)
        std::cout << n << "\n";

    std::cout << l.getBias();

    return std::cout;
}
