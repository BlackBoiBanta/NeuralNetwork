#include "Network.h"
#include <cassert>

namespace ML
{
	NetConfig staticConfig;

	Network::Network()
		:m_layers(staticConfig.getNumLayers())
	{

	}

	DEFINE_ITERATORS(Layer, Network, m_layers);

	uInt Network::size() const
	{
		return m_layers.getLen();
	}

	float Network::getSSR() const
	{
		const Layer& outputLayer = m_layers.back();

		float sum = 0.0f;

		for (const auto& n : outputLayer)
		{
			float residual = getCorrectVal(n.getIndex()) - n.getValue();
			sum += residual * residual;
		}
			
		return sum;

	}

	void Network::setTrueVal(uInt index)
	{
		assert(index < m_layers.back().size());
		m_trueValIndex = index;
	}

	float Network::getCorrectVal(uInt index) const
	{	
		if (index == m_trueValIndex)
			return 1.0f;
		return 0.0f;
	}
}