#include <iostream>
#include "Layer.h"

namespace ML
{
	std::function<mFloat()> setWeight;

	void Layer::init(uInt currentLayerIndex, uInt numLayers, uInt size, Math::FuncPair func, InitMethod init)
	{
		m_func = func;
		m_neurones.resize(size);
		Layer& currentLayer = *this;

		if (init == InitMethod::Xavier_Normal)
		{
			setWeight = [&currentLayer]() -> mFloat
			{
				return nd(mt) * std::pow(1.0f/currentLayer.getBack()->getLen(), 0.5);
			};
		}
		else if (init == InitMethod::Random_Uniform)
		{
			setWeight = [&currentLayer]() -> mFloat
			{
				return urd(mt);
			};
		}


		if (currentLayerIndex == numLayers - 1)
		{
			uInt i = 0;
			for (auto& n : *this)
				n.init(0, i++);
			m_bias.toVar().init(0);

			m_front = nullptr;
		}
		else
			m_front = this + 1;

		if (currentLayerIndex == 0)
			m_back = nullptr;
		else
			m_back = this - 1;

		if (currentLayerIndex > 0)
		{
			uInt i = 0;
			for (auto& n : *getBack())
				n.init(size, i++);
			getBack()->getBias().init(size);
		}
	}
	
	DEFINE_ITERATORS(Neuron, Layer, m_neurones);

	const Layer* Layer::getFront() const
	{
		return m_front;
	}

	const Layer* Layer::getBack() const
	{
		return m_back;
	}

	Layer* Layer::getFront()
	{
		return m_front;
	}

	Layer* Layer::getBack()
	{
		return m_back;
	}

	const Bias& Layer::getBias() const
	{
		return m_bias.toVar();
	}

	Bias& Layer::getBias()
	{
		return m_bias.toVar();
	}

	uInt Layer::getLen() const
	{
		return m_neurones.getLen();
	}

	mFloat Layer::derivFunc(mFloat val) const
	{
		return m_func.DERIV(val);
	}	
	
	mFloat Layer::actFunc(mFloat val) const
	{
		return m_func.ACT(val);
	}

	void Layer::calcDeltaWeights()
	{
		for (Neuron& n : *this)
			n.calcGradient(*getFront());
		getBias().calcGradient(*getFront());
	}

	void Layer::propagateOutputs()
	{
		if (!m_back)
			return;

		for (Neuron& n : *this)
		{
			n.setValue(addOutputs(n.getIndex()));
			n.setOutput(actFunc(n.getValue()));
		}
	}

	void Layer::applyDeltaWeights(mFloat lr, mFloat alpha, uInt noDataValues)
	{
		for (Neuron& n : *this)
			n.applyGradients(lr, alpha, noDataValues);
		getBias().applyGradients(lr, alpha, noDataValues);
	}

	mFloat Layer::addOutputs(uInt index)
	{
		mFloat sum = 0.0f;

		if (!m_back) 
			return sum;

		for (Neuron& n : *m_back)
			sum += n.getWeightedVal(index);

		sum += m_back->getBias()[index].WEIGHT;

		return sum;
	}


}
