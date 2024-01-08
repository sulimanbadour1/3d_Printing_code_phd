#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

using namespace std;

// Specific heat ratio for the fluid (usually air)
const double G0 = 1.4;

// Coefficients derived from the specific heat ratio, used in various fluid dynamics calculations
const double G1 = 0.5 * (G0 - 1) / G0;           // Coefficient for pressure-based calculations
const double G2 = 0.5 * (G0 + 1) / G0;           // Coefficient for entropy and shock wave calculations
const double G3 = 2 * G0 / (G0 - 1);             // Coefficient related to enthalpy calculations
const double G4 = 2 / (G0 - 1);                  // Coefficient for energy and speed calculations
const double G5 = 2 / (G0 + 1);                  // Coefficient for isentropic flow calculations
const double G6 = (G0 - 1) / (G0 + 1);           // Coefficient for flow property relations
const double G7 = (G0 - 1) / 2;                  // Coefficient for velocity calculations
const double G8 = G0 - 1;                        // Coefficient for energy equations
const double G9 = -G2;                           // Negative of G2, used in specific cases
const double G10 = -0.5 * (G0 + 1) / pow(G0, 2); // Coefficient for advanced flow calculations
const double G11 = -0.5 * (3 * G0 + 1) / G0;     // Coefficient for specific heat capacity calculations
const double G12 = 1.0 / G0;                     // Inverse of specific heat ratio

// Courant–Friedrichs–Lewy (CFL) condition for numerical stability in simulations
const double CFL = 0.8;

// Grid parameters for the simulation
const int NumOfPnt = 201;                     // Number of points in the grid
const double xL = 0, xR = 1.0;                // Left and right boundaries of the grid
const double xM = (xL + xR) / 2;              // Midpoint of the grid
const double dx = (xR - xL) / (NumOfPnt - 1); // Distance between grid points

inline double sound_speed(double p, double rho)
{
    return sqrt(G0 * p / rho);
}

inline double internal_energy(double p, double rho)
{
    return p / (G8 * rho);
}

inline double kinetic_energy(double u)
{
    return 0.5 * pow(u, 2);
}

class PrimitiveVar
{
public:
    double rho, u, p;
    double a, e;
    double nbla[3];

public:
    PrimitiveVar(double density = 1.0, double velocity = 0.0, double pressure = 101325.0)
    {
        set(density, velocity, pressure);
    }

    PrimitiveVar(istream &in)
    {
        double density, velocity, pressure;
        in >> density >> velocity >> pressure;
        set(density, velocity, pressure);
    }

    ~PrimitiveVar() = default;

    void set(double density, double velocity, double pressure)
    {
        rho = density;
        u = velocity;
        p = pressure;
        a = sound_speed(p, rho);
        e = internal_energy(p, rho);
        nbla[0] = u - a;
        nbla[1] = u;
        nbla[2] = u + a;
    }

    PrimitiveVar operator+(const PrimitiveVar &rhs)
    {
        return PrimitiveVar(rho + rhs.rho, u + rhs.u, p + rhs.p);
    }
};

inline double Ak(const PrimitiveVar &W)
{
    return G5 / W.rho;
}

inline double Bk(const PrimitiveVar &W)
{
    return G6 * W.p;
}

inline double gk(double p, const PrimitiveVar &W)
{
    return sqrt(Ak(W) / (p + Bk(W)));
}

inline double fk(double p, const PrimitiveVar &W)
{
    if (p > W.p)
        return (p - W.p) * gk(p, W);
    else
        return G4 * W.a * (pow(p / W.p, G1) - 1);
}

inline double dfk(double p, const PrimitiveVar &W)
{
    if (p > W.p)
    {
        double B = Bk(W);
        return sqrt(Ak(W) / (B + p)) * (1 - 0.5 * (p - W.p) / (B + p));
    }
    else
        return 1.0 / (W.rho * W.a) / pow(p / W.p, G2);
}

inline double ddfk(double p, const PrimitiveVar &W)
{
    if (p > W.p)
    {
        double B = Bk(W);
        return -0.25 * sqrt(Ak(W) / (B + p)) * (4 * B + 3 * p + W.p) / pow(B + p, 2);
    }
    else
        return G10 * W.a / pow(W.p, 2) * pow(p / W.p, G11);
}

inline double f(double p, const PrimitiveVar &Wl, const PrimitiveVar &Wr)
{
    return fk(p, Wl) + fk(p, Wr) + (Wr.u - Wl.u);
}

inline double df(double p, const PrimitiveVar &Wl, const PrimitiveVar &Wr)
{
    return dfk(p, Wl) + dfk(p, Wr);
}

inline double ddf(double p, const PrimitiveVar &Wl, const PrimitiveVar &Wr)
{
    return ddfk(p, Wl) + ddfk(p, Wr);
}

// Exact solution of then 1-D Euler equation
// This is the Riemann problem, where initial discontinuity exists
double p_star(const PrimitiveVar &Wl, const PrimitiveVar &Wr)
{
    const double TOL = 1e-6;
    const double du = Wr.u - Wl.u;

    // Select initial pressure
    const double f_min = f(min(Wl.p, Wr.p), Wl, Wr);
    const double f_max = f(max(Wl.p, Wr.p), Wl, Wr);

    double p_m = (Wl.p + Wr.p) / 2;
    p_m = max(TOL, p_m);

    double p_pv = p_m - du * (Wl.rho + Wr.rho) * (Wl.a + Wr.a) / 8;
    p_pv = max(TOL, p_pv);

    const double gL = gk(p_pv, Wl);
    const double gR = gk(p_pv, Wr);
    double p_ts = (gL * Wl.p + gR * Wr.p - du) / (gL + gR);
    p_ts = max(TOL, p_ts);

    double p_tr = pow((Wl.a + Wr.a - G7 * du) / (Wl.a / pow(Wl.p, G1) + Wr.a / pow(Wr.p, G1)), G3);
    p_tr = max(TOL, p_tr);

    double p0 = p_m;
    if (f_min < 0 && f_max < 0)
        p0 = p_ts;
    else if (f_min > 0 && f_max > 0)
        p0 = p_tr;
    else
        p0 = p_pv;

    // Solve
    int iter_cnt = 0;
    double CHA = 1.0;
    while (CHA > TOL)
    {
        ++iter_cnt;

        double fder = df(p0, Wl, Wr);
        if (fder == 0)
            throw "Zero derivative!";

        double fval = f(p0, Wl, Wr);
        double fder2 = ddf(p0, Wl, Wr);
        double p = p0 - fval * fder / (pow(fder, 2) - 0.5 * fval * fder2);
        if (p < 0)
        {
            p0 = TOL;
            break;
        }

        CHA = abs(2 * (p - p0) / (p + p0));
        p0 = p;
    }

    return p0;
}

inline double u_star(double p, const PrimitiveVar &Wl, const PrimitiveVar &Wr)
{
    return (Wl.u + Wr.u) / 2 + (fk(p, Wr) - fk(p, Wl)) / 2;
}

inline double rho_star(double p, const PrimitiveVar &W)
{
    const double t = p / W.p;

    if (t > 1.0)
        return W.rho * ((t + G6) / (G6 * t + 1));
    else
        return W.rho * pow(t, G12);
}

inline double E(const PrimitiveVar &W)
{
    return W.rho * (internal_energy(W.p, W.rho) + kinetic_energy(W.u));
}

inline double E(double density, double velocity, double pressure)
{
    return density * (internal_energy(pressure, density) + kinetic_energy(velocity));
}

inline double H(const PrimitiveVar &W)
{
    return (E(W) + W.p) / W.rho;
}

inline double H(double density, double velocity, double pressure)
{
    return (E(density, velocity, pressure) + pressure) / density;
}

class FluxVar
{
private:
    double f[3];

public:
    FluxVar(double a = 0.0, double b = 0.0, double c = 0.0)
    {
        f[0] = a;
        f[1] = b;
        f[2] = c;
    }

    ~FluxVar() = default;

    void set(const PrimitiveVar &W)
    {
        f[0] = W.rho * W.u;
        f[1] = W.rho * pow(W.u, 2) + W.p;
        f[2] = W.u * (E(W) + W.p);
    }

    void set(const PrimitiveVar &W, bool sign)
    {
        // sign: true for +, false for -
        double lambda[3];
        if (sign)
        {
            lambda[0] = max(W.nbla[0], 0.0);
            lambda[1] = max(W.nbla[1], 0.0);
            lambda[2] = max(W.nbla[2], 0.0);
        }
        else
        {
            lambda[0] = min(W.nbla[0], 0.0);
            lambda[1] = min(W.nbla[1], 0.0);
            lambda[2] = min(W.nbla[2], 0.0);
        }

        f[0] = lambda[0] + 2 * G8 * lambda[1] + lambda[2];
        f[1] = (W.u - W.a) * lambda[0] + 2 * G8 * W.u * lambda[1] + (W.u + W.a) * lambda[2];
        f[2] = (H(W) - W.u * W.a) * lambda[0] + G8 * pow(W.u, 2) * lambda[1] + (H(W) + W.u * W.a) * lambda[2];

        const double tmp = 0.5 * W.rho * G12;
        for (int i = 0; i < 3; i++)
            f[i] *= tmp;
    }

    void set(double density, double velocity, double pressure)
    {
        f[0] = density * velocity;
        f[1] = density * pow(velocity, 2) + pressure;
        f[2] = velocity * (E(density, velocity, pressure) + pressure);
    }

    double &mass_flux()
    {
        return f[0];
    }

    double &momentum_flux()
    {
        return f[1];
    }

    double &energy_flux()
    {
        return f[2];
    }

    FluxVar operator-(const FluxVar &rhs)
    {
        return FluxVar(this->f[0] - rhs.f[0], this->f[1] - rhs.f[1], this->f[2] - rhs.f[2]);
    }

    FluxVar operator+(const FluxVar &rhs)
    {
        return FluxVar(this->f[0] + rhs.f[0], this->f[1] + rhs.f[1], this->f[2] + rhs.f[2]);
    }
};

class ConservativeVar
{
private:
    double u[3];

public:
    ConservativeVar(double a = 0.0, double b = 0.0, double c = 0.0)
    {
        u[0] = a;
        u[1] = b;
        u[2] = c;
    }

    ConservativeVar(const PrimitiveVar &x)
    {
        set(x.rho, x.u, x.p);
    }

    ~ConservativeVar() = default;

    void set(double density, double velocity, double pressure)
    {
        u[0] = density;
        u[1] = density * velocity;
        u[2] = E(density, velocity, pressure);
    }

    double &elem(int idx)
    {
        return u[idx];
    }

    double density() const
    {
        return u[0];
    }

    double velocity() const
    {
        return u[1] / u[0];
    }

    double pressure() const
    {
        return G8 * (u[2] - 0.5 * pow(u[1], 2) / u[0]);
    }

    PrimitiveVar toPrimitive()
    {
        return PrimitiveVar(density(), velocity(), pressure());
    }

    ConservativeVar operator+(const ConservativeVar &rhs)
    {
        return ConservativeVar(this->u[0] + rhs.u[0], this->u[1] + rhs.u[1], this->u[2] + rhs.u[2]);
    }

    ConservativeVar operator-(const ConservativeVar &rhs)
    {
        return ConservativeVar(this->u[0] - rhs.u[0], this->u[1] - rhs.u[1], this->u[2] - rhs.u[2]);
    }
};

void W_fanL(const PrimitiveVar *W, double S, PrimitiveVar &ans)
{
    double density = W->rho * pow(G5 + G6 / W->a * (W->u - S), G4);
    double velocity = G5 * (W->a + G7 * W->u + S);
    double pressure = W->p * pow(G5 + G6 / W->a * (W->u - S), G3);
    ans.set(density, velocity, pressure);
}

void W_fanR(const PrimitiveVar *W, double S, PrimitiveVar &ans)
{
    double density = W->rho * pow(G5 - G6 / W->a * (W->u - S), G4);
    double velocity = G5 * (-W->a + G7 * W->u + S);
    double pressure = W->p * pow(G5 - G6 / W->a * (W->u - S), G3);
    ans.set(density, velocity, pressure);
}

void output(ofstream &f, int time_step, const vector<PrimitiveVar> &W, const vector<PrimitiveVar> &W_exact)
{
    f << time_step << endl;
    for (int i = 1; i <= NumOfPnt; i++)
    {
        f << W[i].rho << '\t' << W[i].u << '\t' << W[i].p << '\t' << W[i].e;
        f << '\t' << W_exact[i].rho << '\t' << W_exact[i].u << '\t' << W_exact[i].p << '\t' << W_exact[i].e << endl;
    }
}

void ExactSol(const PrimitiveVar &Wl, const PrimitiveVar &Wr, double p_s, double u_s, double rho_sL, double rho_sR, double S, PrimitiveVar &ans)
{
    if (S < u_s)
    {
        const double S_L = Wl.u - Wl.a * sqrt(G2 * p_s / Wl.p + G1);
        const double S_HL = Wl.u - Wl.a;
        const double S_TL = u_s - sound_speed(p_s, rho_sL);

        if (p_s > Wl.p)
        {
            if (S < S_L)
                ans = Wl;
            else
                ans.set(rho_sL, u_s, p_s);
        }
        else
        {
            if (S < S_HL)
                ans = Wl;
            else if (S > S_TL)
                ans.set(rho_sL, u_s, p_s);
            else
                W_fanL(&Wl, S, ans);
        }
    }
    else
    {
        const double S_R = Wr.u + Wr.a * sqrt(G2 * p_s / Wr.p + G1);
        const double S_HR = Wr.u + Wr.a;
        const double S_TR = u_s + sound_speed(p_s, rho_sR);

        if (p_s > Wr.p)
        {
            if (S > S_R)
                ans = Wr;
            else
                ans.set(rho_sR, u_s, p_s);
        }
        else
        {
            if (S > S_HR)
                ans = Wr;
            else if (S < S_TR)
                ans.set(rho_sR, u_s, p_s);
            else
                W_fanR(&Wr, S, ans);
        }
    }
}

int main(int argc, char **argv)
{
    // Coordinates
    vector<double> x(NumOfPnt, xL);
    for (int k = 1; k < NumOfPnt; ++k)
        x[k] = x[k - 1] + dx;

    // Loop all cases
    int NumOfCase;
    cin >> NumOfCase;
    for (int c = 0; c < NumOfCase; ++c)
    {
        // Input
        PrimitiveVar Wl(cin), Wr(cin);
        int NumOfStep;
        cin >> NumOfStep;

        // For exact solution
        const double p_middle = p_star(Wl, Wr);
        const double u_middle = u_star(p_middle, Wl, Wr);
        const double rho_middle_left = rho_star(p_middle, Wl);
        const double rho_middle_right = rho_star(p_middle, Wr);

        // Create output data file
        stringstream ss;
        ss << c + 1;
        string header = "Case" + ss.str() + ".txt";
        ofstream fout(header);
        if (!fout)
            throw "Failed to open file!";

        // Output coordinates and intial settings
        fout << NumOfStep << '\t' << NumOfPnt << endl;
        for (int i = 0; i < NumOfPnt; i++)
            fout << x[i] << endl;

        // Initialize
        int PREV = 0, CUR = 1;
        vector<vector<PrimitiveVar>> w(2, vector<PrimitiveVar>(NumOfPnt + 2));
        w[PREV][0] = Wl;
        for (int j = 1; j <= NumOfPnt; ++j)
        {
            if (x[j - 1] < xM)
                w[PREV][j] = Wl;
            else
                w[PREV][j] = Wr;
        }
        w[PREV][NumOfPnt + 1] = Wr;
        output(fout, 0, w[PREV], w[PREV]);

        // Iterate
        double t = 0.0;
        for (int i = 1; i < NumOfStep; i++)
        {
            // Choose proper time-step
            double S = 0.0;
            for (int j = 1; j <= NumOfPnt; ++j)
            {
                double S_local = fabs(w[PREV][j].u) + w[PREV][j].a;
                if (S_local > S)
                    S = S_local;
            }
            const double dt = CFL * dx / S;
            t += dt;

            // Flux Split according to eigen-value at each pnt
            vector<vector<FluxVar>> f(2, vector<FluxVar>(NumOfPnt + 2));
            for (int j = 0; j < NumOfPnt + 2; ++j)
            {
                f[0][j].set(w[PREV][j], false);
                f[1][j].set(w[PREV][j], true);
            }

            // Calculate intercell flux
            vector<FluxVar> mf(NumOfPnt + 1);
            for (int j = 0; j <= NumOfPnt; ++j)
                mf[j] = f[1][j] + f[0][j + 1];

            // Apply the Godunov method
            const double r = -dt / dx;
            for (int j = 1; j <= NumOfPnt; ++j)
            {
                FluxVar cf = mf[j] - mf[j - 1];
                ConservativeVar dU(cf.mass_flux() * r, cf.momentum_flux() * r, cf.energy_flux() * r);
                ConservativeVar U_prev(w[PREV][j]);
                ConservativeVar U_cur = U_prev + dU;
                w[CUR][j] = U_cur.toPrimitive();
            }

            // Transmissive B.C.
            w[CUR][0] = w[CUR][1];
            w[CUR][NumOfPnt + 1] = w[CUR][NumOfPnt];

            // Calculate the exact solution for comparision
            vector<PrimitiveVar> w_exact(NumOfPnt + 2);
            for (int j = 1; j <= NumOfPnt; ++j)
            {
                const double S = (x[j - 1] - xM) / t;
                ExactSol(Wl, Wr, p_middle, u_middle, rho_middle_left, rho_middle_right, S, w_exact[j]);
            }

            // Log
            output(fout, i, w[CUR], w_exact);
            swap(PREV, CUR);
        }
        fout.close();
    }

    return 0;
}
