#pragma once 
#include "Training/TrainingVars.h"
#include "Math/Math.h"
#include <random>
#include "NetConfig.h"
#include "Data/Data.h"

#define DECLARE_ITERATORS(x, y) x* end(); x* begin(); const x* end()const; const x* begin()const; x& operator[](uInt); const x& operator[](uInt)const;
#define DEFINE_ITERATORS(x, y, vec) x* y::end(){return vec.end();} x* y::begin(){return vec.begin();} const x* y::end() const{return vec.end();} const x* y::begin() const{return vec.begin();} const x& y::operator[](uInt i)const{return vec[i];} x& y::operator[](uInt i){return vec[i];}

namespace ML 
{
  extern NetConfig saticConfig;

  struct Connection
  {
      float WEIGHT = rand() / RAND_MAX;
      float DELTA_WEIGHT = 0.0f;
  };

  class Node
  {
  public:
      Node();
      Node(const Node&) = delete;

      virtual ~Node() = default;      
      
      Connection& operator[](uInt);
      const Connection& operator[](uInt i) const;
      
      Connection* end();
      Connection* begin();
      
      uInt getNumConnections();
      float getBiasVal(uInt nIndex) const;

  protected:
      Vec<Connection> m_connections;

  };
  
  class Neurone : public Node
  {
    public:
      static uInt sIndex;

      Neurone();
      Neurone(const Neurone&) = delete;

      void setValue(float);
      float getIndex() const;
      float getValue() const;
      
      void setGradient(float val);              // training
      float getWeightedVal(Neurone&);           // training
      float getGradient() const;                // training
      void calcGradient(Layer& nextLayer);      // training
  private:
      uInt m_index;
      NEURONE_T_VARS;

      friend Node;
  };
  

}