#include "Config.h"
#include <cassert>


namespace ML
{
	NetConfig cfg;

	NetConfig::NetConfig(const uInt* topology, uInt size, mFloat learningRate, mFloat alpha)
		:Config(topology, size, learningRate, alpha)
	{
	}

	NetConfig::NetConfig(const Vec<uInt>& topology, mFloat learningRate, mFloat alpha)
		:Config(topology.begin(), topology.getLen(), learningRate, alpha)
	{
	}

	void NetConfig::setLR(mFloat val)
	{
		m_learningRate = val;
	}

	void NetConfig::setAlpha(mFloat val)
	{
		m_alpha = val;
	}

	uInt NetConfig::getTrueVal() const
	{
		return m_trueValIndex;
	}

	void NetConfig::setTrueVal(uInt index)
	{
		m_trueValIndex = index;
	}

	void NetConfig::setAllActivationFuncs(Math::FuncPair pair)
	{
		for (uInt i = 0; i < getNumLayers(); ++i)
			m_layerConfig[i].func = pair;
	}

	void NetConfig::setLayerActivationFunc(Math::FuncPair pair, uInt index)
	{
		assert(index < getNumLayers());
		m_layerConfig[index].func = pair;
	}

	void NetConfig::setLayerInitMethod(InitMethod method, uInt index)
	{
		assert(index < getNumLayers());
		m_layerConfig[index].init = method;

	}

	void NetConfig::setAllInitMethods(InitMethod method)
	{
		for (uInt i = 0; i < getNumLayers(); ++i)
			m_layerConfig[i].init = method;
	}
}
