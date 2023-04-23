#include "Network.h"
#include <cmath>

namespace ML
{	
	Network::Network(const NetConfig& config)
		: m_layers(config.getNumLayers()), m_config(config), m_numberOfDataValues(0)
	{
		assert(config.getLR() != 0.0f);
		assert(config.getNumLayers() != 0);

		uInt i = 0;
		for (Layer& l : *this)
		{
			l.init(i, config.getNumLayers(), config[i].len, config[i].func, config[i].init);
			i++;
		}
	}

	DEFINE_ITERATORS(Layer, Network, m_layers);

	void Network::backProp()
	{
		for (Layer* cl = getOutputLayer().getBack(); cl; cl = cl->getBack())
			cl->applyDeltaWeights(m_config.getLR(), m_config.getAlpha(), m_numberOfDataValues);

		m_numberOfDataValues = 0;
	}


	Layer& Network::getInputLayer()
	{
		return (*this)[0];
	}

	Layer& Network::getOutputLayer()
	{
		return (*this)[getLen() - 1];
	}	
	
	const Layer& Network::getInputLayer() const
	{
		return (*this)[0];
	}

	const Layer& Network::getOutputLayer() const
	{
		return (*this)[getLen() - 1];
	}

	void Network::calculate()
	{
		++m_numberOfDataValues;

		for (Neurone& n : getOutputLayer())
		{
			mFloat derivative = -2.0f;
			derivative *= getCorrectVal(n.getIndex()) - n.getOutput();
			derivative *= getOutputLayer().derivFunc(n.getValue());

			n.setGradient(derivative);
		}

		for (Layer* cl = getOutputLayer().getBack(); cl; cl = cl->getBack())
			cl->calcDeltaWeights();

	}

	uInt Network::getLen() const
	{
		return m_config.getNumLayers();
	}

	mFloat Network::getSSR() const
	{
		const Layer& outputLayer = m_layers.back();

		mFloat sum = 0.0f;

		for (const auto& n : outputLayer)
		{
			mFloat residual = getCorrectVal(n.getIndex()) - n.getOutput();
			sum += residual * residual;
		}

		return std::pow(sum, 0.5f);

	}
	void Network::setCorrectVal(uInt index, mFloat val)
	{
		m_trueVal = val;
		m_config.setTrueVal(index);
	}

	void Network::propagate()
	{
		for (Layer& l : *this)
			l.propagateOutputs();
	}

	mFloat Network::getCorrectVal(uInt index) const
	{
		if (index == m_config.getTrueVal())
			return m_trueVal;
		return 0.0f;
	}


}



