#include "NetConfig.h"
#include <cassert>

namespace ML
{
	NetConfig:: NetConfig(uInt* layers, uInt size, float eta, float alpha, float learningRate)
		:m_eta(eta), m_alpha(alpha), m_learningRate(learningRate), m_layerConfig(size), m_layerIndex(0)
	{
		for (uInt i = 0; i < size; ++i)
			m_layerConfig[i].size = layers[i];
	}
	
	uInt NetConfig::getNumLayers() const
	{
		return m_layerConfig.getLen();
	}

	uInt NetConfig::sampleNeuroneNum() 
	{
		return m_layerConfig[m_layerIndex++].size;
	}

	void NetConfig::setAllActivationFuncs(Math::FuncPair pair)
	{
		for (uInt i = 0; i < this->getNumLayers(); ++i)
			m_layerConfig[i].func = pair;
	}

	void NetConfig::setLayerActivationFunc(Math::FuncPair pair, uInt index)
	{
		assert(index < getNumLayers());
		m_layerConfig[index].func = pair;
	}










}