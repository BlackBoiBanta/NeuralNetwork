#pragma once 

#include <iostream>
#include <functional>
#include <random>
#include "Config/Config.h"

namespace ML 
{
  extern std::mt19937 mt; 
  extern std::normal_distribution<mFloat> nd;
  extern std::uniform_real_distribution<mFloat> urd;

  extern std::function<mFloat()> setWeight;

  template <uInt MAX>
  struct Connection
  {
      mFloat DERIVATIVE = 0.0f;
      mFloat OLD_DERIVATIVE = 0.0f;
      mFloat WEIGHT = setWeight();

      void update(mFloat lr, mFloat alpha, uInt noDataValues)
      {
          WEIGHT -= (DERIVATIVE * lr + alpha * OLD_DERIVATIVE) / static_cast<mFloat>(noDataValues);
          OLD_DERIVATIVE = DERIVATIVE;
          DERIVATIVE = 0.0f;
      }
  };

  class Layer;

  class Bias
  {
  public:
      Bias() = default;

      void init(uInt numConnections);
      
      void calcGradient(const Layer& nextLayer);
      void applyGradients(mFloat lr, mFloat alpha, uInt noDataValues);

      DECLARE_ITERATORS(Connection<0>, Bias);

  private:
      Vec<Connection<0>> m_connections;
  }; 

  class Neurone
  {
  public:

      Neurone() = default;

      void init(uInt numConnections, uInt index);

      void setOutput(mFloat);
      mFloat getOutput() const;

      void setValue(mFloat);
      mFloat getValue() const;
      
      void setGradient(mFloat);
      mFloat getGradient() const;
      
      mFloat getWeightedVal(uInt) const;
      
      uInt getIndex() const;

      void calcGradient(const Layer& nextLayer);
      void applyGradients(mFloat lr, mFloat alpha, uInt noDataValues);

      DECLARE_ITERATORS(Connection<2>, Neurone);

      friend class Network;

  private:

      uInt m_index;
      mFloat m_value;
      mFloat m_output;
      mFloat m_gradient;

      Vec<Connection<2>> m_connections;

  };
}

template<ML::uInt max>
inline std::ostream& operator<<(std::ostream&, const ML::Connection<max> cn)
{
    std::cout << "WEIGHT\t->\t" << cn.WEIGHT << "\nDERIVATIVE\t->\t" << cn.DERIVATIVE << "\nOLD_DERIVATIVE\t->\t" << cn.OLD_DERIVATIVE << "\n";
    return std::cout;
}


inline std::ostream& operator<<(std::ostream&, const ML::Neurone& n)
{
    std::cout << "==================\tNeurone index " << n.getIndex() << "\t==================\n\n";
    std::cout << "Value\t->\t" << n.getValue() << "\nOutput\t->\t" << n.getOutput() << "\n";

    for (const auto& cn : n)
        std::cout << cn << "\n";

    return std::cout;
}

inline std::ostream& operator<<(std::ostream&, const ML::Bias& n)
{
    std::cout << "======\tBias\t=====\n\n";

    for (const auto& cn : n)
        std::cout << cn << "\n";

    return std::cout;
}
