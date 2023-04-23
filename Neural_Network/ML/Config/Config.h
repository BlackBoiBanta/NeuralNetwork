#pragma once
#include <string>
#include "Data/Data.h"
#include "Math/Math.h"

namespace ML
{
	// make another class for netconfig that takes in a file path in its ctor

	enum class InitMethod;

	struct LayerData
	{
		uInt len;
		Math::FuncPair func;
		InitMethod init;
	};

	class Config
	{
	
	public:
		mFloat getLR() const;
		mFloat getAlpha() const;
		uInt getNumLayers() const;


		void incrementCurrentLayer();
		uInt currentLayerIndex() const;

		uInt getCurrentNeuroneIndex();
		void resetCurrentNeuroneIndex();

		DECLARE_ITERATORS(LayerData, Config);

	protected:
		Config();
		Config(const uInt*, uInt, mFloat learningRate, mFloat alpha);

		virtual ~Config() = default;

		mFloat m_alpha;
		mFloat m_learningRate;

		Vec<LayerData> m_layerConfig;

		uInt m_currentNeuroneIndex;
		uInt m_currentLayer = 0;

	};

	class NetConfig : public Config
	{
	public:
	
		NetConfig() = default;
		NetConfig(const uInt*, uInt, mFloat learningRate, mFloat alpha);					// give a vector of the number of neurones in each layer
		NetConfig(const Vec<uInt>&, mFloat learningRate, mFloat alpha);
		
		void setLR(mFloat);
		void setAlpha(mFloat);

		void setAllActivationFuncs(Math::FuncPair);
		void setLayerActivationFunc(Math::FuncPair, uInt);

		void setAllInitMethods(InitMethod method);
		void setLayerInitMethod(InitMethod method, uInt index);

		friend class Network;

	private:
		uInt getTrueVal() const;
		void setTrueVal(uInt index);
		uInt m_trueValIndex;

	};

}