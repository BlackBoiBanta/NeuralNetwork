#pragma once

#include <cstdint>
#include <initializer_list>
#include <cassert>

#define DECLARE_ITERATORS(x, y) x* end(); x* begin(); const x* end()const; const x* begin()const; x& operator[](uInt); const x& operator[](uInt)const;
#define DEFINE_ITERATORS(x, y, vec) x* y::end(){return vec.end();} x* y::begin(){return vec.begin();} const x* y::end() const{return vec.end();} const x* y::begin() const{return vec.begin();} const x& y::operator[](uInt i)const{assert(i < vec.getLen()); return vec[i];} x& y::operator[](uInt i){ assert(i < vec.getLen()); return vec[i];}

namespace ML
{
    using uInt = std::uint32_t;;
    using mFloat = float;

    template <typename Dat>
    class Vec
    {
    public:
        Vec() = default;

        Vec(uInt len)
            :m_data(new Dat[len]), m_len(len)
        {
        }

        Vec(const std::initializer_list<Dat>& list)
            :m_data(new Dat[list.size()]), m_len(list.size())
        {
            uInt i = 0;
            for (const auto& element : list)
                m_data[i++] = element;
        }

        Vec(const Vec& copy)
        {
            *this = copy;
        }

        Vec(Vec&& move)
            :m_len(move.m_len), m_data(move.m_data)
        {
            move.m_data = nullptr;
            move.m_len = 0;
        }

        virtual ~Vec()
        {
            if (m_data)
                delete[] m_data;
        }

        Vec& operator=(const Vec& copy)
        {
            m_len = copy.m_len;
            m_data = new Dat[copy.m_len];

            uInt i = 0;
            for (const auto& el : copy)
                (*this)[i++] = el;
            
            return *this;
        }

        Dat& operator[](uInt i)
        {
            assert(i < m_len);

            return m_data[i];
        }

        const Dat& operator[](uInt i) const
        {
            assert(i < m_len);

            return m_data[i];
        }

        Dat* begin()
        {
            return m_data;
        }

        Dat* end()
        {
            return m_data + m_len;
        }        
        
        const Dat* begin() const
        {
            return m_data;
        }

        const Dat* end() const
        {
            return m_data + m_len;
        }

        Dat* data()
        {
            return m_data;
        }

        const Dat* data() const
        {
            return m_data;
        }

        uInt getLen() const
        {
            return m_len;
        }

        void resize(uInt newSize) // once resized, you can't get back the values 
        {
            if (newSize == m_len)
            {
                return;
            }
            else if (!newSize)
            {
                if (m_data)
                    delete[] m_data;

                m_len = 0;
                m_data = nullptr;
            }
            else if (newSize > m_len)
            {
                Vec<Dat> newVec(newSize);

                uInt i = 0;
                for (auto& el : *this)
                    newVec[i++] = el;

                *this = newVec;
            }
            else if (newSize < m_len)
                m_len = newSize;
        }

        const Dat& back() const 
        {
            return m_data[m_len - 1];
        }        
        
        Dat& back()
        {
            return m_data[m_len - 1];
        }

    private:
        uInt m_len = 0;

    protected:
        Dat* m_data = nullptr;

    };
 
    template <typename Dat>
    class Var : Vec<Dat> 
    {
    public:

        using Vec<Dat>::m_data;

        Var(uInt num = 1)
            :Vec<Dat>(num == 0 ? 0 : 1)
        {
        }

        Var(const Var& copy)
            :Vec<Dat>(1)
        {
            *m_data = *copy.m_data;
        }

        Var(Var&& move)
            :m_data(move.m_data)
        {
            move.m_data = nullptr;
        }

        Dat& toVar()
        {
            return *m_data;
        }

        const Dat& toVar() const
        {
            return *m_data;
        }

    };

 
}