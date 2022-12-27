
//#pragma once 
//
//#include <vector>
//#include <string>
//#include <cassert>
//
//#include "Math/Math.h"
//#include "Data.h"
//
//#define ASSERT_LAYER_VAL(x) assert(x != 0 && "Invalid number for "#x);
//
//namespace ML
//{
//
//  class Neurone;
//  class NetConfig;
//  
//  extern NetConfig sConfig;
//  extern uInt sLayerNCount;
//
//  typedef struct
//  {
//      std::string LayerName;
//      Math::FuncPair func;
//  }FuncKey;
//
//  typedef struct
//  {
//      std::string outputName;
//      float value;
//  }OutputVal;
//
//  class NetConfig  
//  {
//  private:
//
//    uInt nINPUT_LAYER;
//    uInt nHIDDEN_LAYER;
//    uInt nOUTPUT_LAYER;
//
//    uInt NUM_HIDDEN;
//
//    float PASS_VAL;
//    float LEARNING_RATE;
//    std::string TRUE_VAL;
//    std::vector<FuncKey> FUNC_MAP;
//    
//    void setTrueVal(const std::string&);
//    float getNTrueValue(const Neurone&);
//
//  public:
//    
//    NetConfig() = default;
//    NetConfig(uInt nInput, uInt nHidden, uInt nOutput, uInt hiddenNum, float lr = 0.1f, Math::FuncPair actFunc = Math::softPlusFunc);
//    
//    void setPassVal(float);
//    void initOutput(const std::vector<std::string>&);
//    Math::FuncPair& operator[](uInt index);
//    Math::FuncPair& operator[](const std::string& key);
//    
//    std::vector<OutputVal> OUTPUT_VEC;
//    
//    std::size_t getNumLayers()
//    {
//        return FUNC_MAP.size();
//    }  
//    friend class Layer;
//    friend class Network;
//   
//  };
//
//
//
//
//}

#include "Data/Data.h"
#include "Math/Math.h"

namespace ML
{
	struct LayerData
	{
		uInt size;
		Math::FuncPair func;
	};

	class NetConfig
	{
	public:
		NetConfig(uInt*, uInt, float, float, float);					// give a vector of the number of neurones in each layer
		NetConfig(const NetConfig&) = default;

		uInt sampleNeuroneNum();
		uInt getNumLayers() const;
		void setTrueVal(uInt index);
		void setAllActivationFuncs(Math::FuncPair);
		void setLayerActivationFunc(Math::FuncPair, uInt);

	private:
		float m_eta;
		float m_alpha;
		float m_learningRate;
		float m_trueValIndex;
		Vec<LayerData> m_layerConfig = Vec<LayerData>(0);
		
		uInt m_layerIndex;
		NetConfig() = default;
		friend class Network;
	};

}