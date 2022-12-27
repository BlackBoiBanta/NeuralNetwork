#pragma once 

#include "Neurone.h"

namespace ML
{
  class Layer
  {
    public:
      
      Layer();

      void propagateOutputs();
      float addOutputs(Neurone&);
      
      DECLARE_ITERATORS(Neurone, Layer);

      uInt size() const;
      const Node& getBias();
      
      void backProp();                                                           // training
      static void calcDeltaWeights(Layer*, float chain = 0.0f, uInt index = 0);  // training
    
  private:
      Math::FuncPair m_func;
      Vec<Neurone> m_neurones;
      Var<Node> m_bias;

      // in layer c_tor, set the static nIndex to zero each time c_tor is called. so smart. and then 
      // it seems smarter to just make the layers hold the layer poiters instead, as it's quite weird for
      // a contained class to hold a pointer to its own container class.
  };

}