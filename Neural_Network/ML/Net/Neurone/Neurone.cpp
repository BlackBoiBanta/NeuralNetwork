#include <iostream>
#include "Net/Layer/Layer.h"

namespace ML
{
	std::mt19937 mt{};
	std::normal_distribution<mFloat> nd{ 0, 1 };
	std::uniform_real_distribution<mFloat> urd{ -1, 1 };

	void Neurone::init(uInt numConnections, uInt index)
	{
		m_index = index;
		m_connections.resize(numConnections);
	}

	DEFINE_ITERATORS(Connection<2>, Neurone, m_connections);

	uInt Neurone::getIndex() const
	{
		return m_index;
	}

	void Neurone::setOutput(mFloat out)
	{
		m_output = out;
	}

	void Neurone::setValue(mFloat val)
	{
		m_value = val;
	}



	mFloat Neurone::getOutput() const
	{
		return m_output;
	}

	mFloat Neurone::getValue() const
	{
		return m_value;
	}
		


	mFloat Neurone::getGradient() const
	{
		return m_gradient;
	}

	void Neurone::setGradient(mFloat grad)
	{
		m_gradient = grad;
	}


	void Neurone::applyGradients(mFloat lr, mFloat alpha, uInt noDataValues)
	{
		for (auto& cn : *this)
			cn.update(lr, alpha, noDataValues);
	}

	mFloat Neurone::getWeightedVal(uInt index) const
	{
		return getOutput() * (*this)[index].WEIGHT;
	}

	void Neurone::calcGradient(const Layer& nextLayer)
	{
		m_gradient = 0.0f;
		const Layer& thisLayer = *nextLayer.getBack();
		
		for (const Neurone& n : nextLayer)
		{
			uInt nIndex = n.getIndex();
			
			m_gradient += n.getGradient() * (*this)[nIndex].WEIGHT; 
			// get the gradient of the neurone's m_output, with respect to the 
			// loss function, and store as m_gradient
		
			(*this)[nIndex].DERIVATIVE += n.getGradient() * getOutput();

			// calculate derivative of each weight in this neurone, 
			// with respect to loss function
		}

		m_gradient *= thisLayer.derivFunc(getValue());
	}
}



