#include "Layer.h"

namespace ML
{
  Layer::Layer()
      :m_neurones(
  {
  }

  uInt Layer::size() const
  {
      return m_neurones.getLen();
  }

  float Layer::addOutputs(Neurone& neurone)
  {
      float sum = 0.0f;
      
      if (!this)
          return sum;

      for (Neurone& n : *this)
          sum += n.getWeightedVal(neurone);

      sum += (*m_bias).getBiasVal(neurone.getIndex());

      return sum;
  }

  const Node& Layer::getBias()
  {
      return m_bias;
  }

  DEFINE_ITERATORS(Neurone, Layer, m_neurones);

}