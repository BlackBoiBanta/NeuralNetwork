#include "Config.h"

namespace ML
{
	Config::Config(const uInt* layers, uInt size, mFloat learningRate, mFloat alpha)
		:m_alpha(alpha), m_learningRate(learningRate), m_layerConfig(size), m_currentNeuroneIndex(0)
	{
		for (uInt i = 0; i < size; ++i)
		{
			m_layerConfig[i].len = layers[i];
			m_layerConfig[i].func = Math::sigmoidFunc;
		}
	}

	Config::Config()
		:m_layerConfig(0)
	{

	}

	DEFINE_ITERATORS(LayerData, Config, m_layerConfig);


	uInt Config::getNumLayers() const
	{
		return m_layerConfig.getLen();
	}

	mFloat Config::getAlpha() const
	{
		return m_alpha;
	}

	mFloat Config::getLR() const
	{
		return m_learningRate;
	}

}