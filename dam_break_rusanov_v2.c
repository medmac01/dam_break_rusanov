#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>


#define NX     100001
#define G      1.0
#define CFL    0.9
#define XLEFT -10.0
#define XRIGHT 10.0
#define XM     0.0
#define HLEFT  5.0
#define HRIGHT 2.0
#define TEND   3.0

// Initialisation des conditions (discontinues)
void init_conditions(int nc, double *x, double *h, double *u) {
    for (int i = 0; i < nc; ++i) {
        if (x[i] < XM)
            h[i] = HLEFT;
        else
            h[i] = HRIGHT;
        u[i] = 0.0;
    }
}

// Flux de Rusanov
void rusanov_flux(int nc, double *h, double *u, double flux[2][NX - 1]) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < nc - 1; ++i) {
        double hL = h[i], hR = h[i + 1];
        double uL = u[i], uR = u[i + 1];

        double WL[2] = {hL, hL * uL};
        double WR[2] = {hR, hR * uR};

        double FL_L[2] = {hL * uL, hL * uL * uL + 0.5 * G * hL * hL};
        double FL_R[2] = {hR * uR, hR * uR * uR + 0.5 * G * hR * hR};

        double sL = fabs(uL) + sqrt(G * hL);
        double sR = fabs(uR) + sqrt(G * hR);
        double smax = fmax(sL, sR);

        for (int j = 0; j < 2; ++j)
            flux[j][i] = 0.5 * (FL_L[j] + FL_R[j]) - 0.5 * smax * (WR[j] - WL[j]);
    }
}

int main() {
    int nx = NX;
    int nc = nx - 1;

    double start, end;
    double dx = (XRIGHT - XLEFT) / (nx - 1);
    double x[NX], xc[NX - 1];
    double h[NX - 1], u[NX - 1], hn[NX - 1], un[NX - 1];
    double W[2][NX - 1], Wn[2][NX - 1], flux[2][NX - 1];

    start = omp_get_wtime();
    // start = (double)clock() / CLOCKS_PER_SEC;
    // Grille et centres
    for (int i = 0; i < nx; ++i)
        x[i] = XLEFT + i * dx;
    for (int i = 0; i < nc; ++i)
        xc[i] = 0.5 * (x[i] + x[i + 1]);

    // Initialisation
    init_conditions(nc, x, h, u);
    for (int i = 0; i < nc; ++i) {
        W[0][i] = h[i];
        W[1][i] = h[i] * u[i];
    }

    double t = 0.0;
    while (t < TEND) {
        // Vitesse u et célérité
        double max_speed = 0.0;
        #pragma omp parallel for schedule(static) reduction(max:max_speed)
        for (int i = 0; i < nc; ++i) {
            h[i] = W[0][i];
            u[i] = W[1][i] / (h[i] > 1e-6 ? h[i] : 1e-6);
            double c = sqrt(G * h[i]);
            double speed = fabs(u[i]) + c;
            if (speed > max_speed)
                max_speed = speed;
        }

        // Pas de temps
        double dt = CFL * dx / max_speed;
        if (t + dt > TEND)
            dt = TEND - t;
        double nu = dt / dx;
        t += dt;

        // Calcul des flux
        
        rusanov_flux(nc, h, u, flux);
    
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < nc - 1; ++i) {
            for (int j = 0; j < 2; ++j)
                Wn[j][i] = W[j][i] - nu * (flux[j][i] - flux[j][i - 1]);
        }

        // Conditions aux bords (Neumann)
        for (int j = 0; j < 2; ++j) {
            Wn[j][0] = Wn[j][1];
            Wn[j][nc - 1] = Wn[j][nc - 2];
            printf("Wn[%d][0] = %f, Wn[%d][%d] = %f\n", j, Wn[j][0], j, nc - 1, Wn[j][nc - 1]);
        }
        

        // Mise à jour du tableau principal
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nc; ++i) {
            W[0][i] = Wn[0][i];
            W[1][i] = Wn[1][i];
        }

        // Affichage terminal (optionnel)
        printf("t = %.4f\n", t);
    }

    // Sortie finale dans un fichier
    FILE *f = fopen("output_par.dat", "w");
    for (int i = 0; i < nc; ++i)
      fprintf(f, "%f\t%f\t%f\n", xc[i], W[0][i], W[1][i]/W[0][i]);
    fclose(f);
    end = omp_get_wtime();
    // end = (double)clock() / CLOCKS_PER_SEC;

    printf("Simulation terminée. Résultats écrits dans output.dat\n");
    printf("Temps d'exécution: %.4f secondes, Rapport de parallélisation: %.4f\n", end - start, end / start);
    return 0;
}
