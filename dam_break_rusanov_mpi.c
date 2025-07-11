#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h> // MPI header

// Constants
#define NX     100001 // Total number of grid points
#define G      1.0
#define CFL    0.9
#define XLEFT -10.0
#define XRIGHT 10.0
#define XM     0.0    // Midpoint for initial condition
#define HLEFT  5.0    // Initial height left of XM
#define HRIGHT 2.0    // Initial height right of XM
#define TEND   3.0    // Simulation end time


void init_conditions_local(int nc_local, double *xc_local, double *h_local, double *u_local,
                           int global_start_idx, double dx, double XLEFT_GLOBAL, double XM_GLOBAL) {
    for (int i = 0; i < nc_local; ++i) { // Boucle sur les indices de cellules locales (0 à nc_local-1)
        // Calcule la coordonnée x globale de l'INTERFACE GAUCHE de cette cellule
        double current_x_left_interface_global = XLEFT_GLOBAL + (global_start_idx + i) * dx;
        
        // Stocke la coordonnée x du CENTRE de la cellule pour l'affichage (c'est toujours ce qu'on veut en sortie)
        xc_local[i] = current_x_left_interface_global + 0.5 * dx; 

        // Application des conditions initiales, basée sur l'INTERFACE GAUCHE, pour correspondre au code séquentiel
        if (current_x_left_interface_global < XM_GLOBAL) // <-- MODIFICATION ICI : utilise l'interface gauche
            h_local[i + 1] = HLEFT; // +1 pour décaler à cause de la cellule halo gauche
        else
            h_local[i + 1] = HRIGHT; // +1 pour décaler à cause de la cellule halo gauche
        u_local[i + 1] = 0.0; // +1 pour décaler à cause de la cellule halo gauche
    }
}

// Calcul du flux de Rusanov
void rusanov_flux(int nc_local, double *h_local, double *u_local, double flux[2][nc_local + 1]) {
    for (int i = 0; i < nc_local + 1; ++i) { // i correspond à l'index de l'interface
        // hL/uL sont les valeurs de la cellule à gauche de l'interface 'i'
        // hR/uR sont les valeurs de la cellule à droite de l'interface 'i'
        double hL = h_local[i];
        double hR = h_local[i + 1];
        double uL = u_local[i];
        double uR = u_local[i + 1];

        // Gérer les fonds secs ou très faibles hauteurs pour la stabilité numérique
        if (hL < 1e-6) { hL = 1e-6; uL = 0.0; }
        if (hR < 1e-6) { hR = 1e-6; uR = 0.0; }

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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv); // Initialisation de MPI

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Récupération du rang du processus
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); // Récupération du nombre total de processus

    printf("Rank %d of %d processes\n", rank, num_procs);

    int nx_total = NX;
    int nc_total = nx_total - 1; // Nombre total de cellules dans le domaine global

    // Décomposition du domaine : calcul du nombre de cellules locales par processus
    int nc_base = nc_total / num_procs;
    int remainder = nc_total % num_procs;
    int nc_local = nc_base + (rank < remainder ? 1 : 0); // Chaque processus reçoit au moins nc_base cellules,
                                                          // et les 'remainder' premiers processus en reçoivent une de plus.

    // Calcul de l'indice de début global pour les cellules de ce processus
    int global_start_idx = rank * nc_base + fmin(rank, remainder);

    // Calcul du pas d'espace (dx)
    double dx = (XRIGHT - XLEFT) / (nx_total - 1);

    // Allocation des tableaux locaux. Les tableaux de variables d'état (h, u, W, Wn)
    double *xc_local = (double *)malloc(nc_local * sizeof(double)); // Coordonnées x des centres des cellules locales (sans halo)

    double *h_local = (double *)malloc((nc_local + 2) * sizeof(double)); 
    double *u_local = (double *)malloc((nc_local + 2) * sizeof(double)); 

    double W_local[2][nc_local + 2];  // [0]=h, [1]=h*u
    double Wn_local[2][nc_local + 2]; 

    // Tableau de flux : nc_local + 1 interfaces pour nc_local + 2 cellules (halos inclus)
    double flux_local[2][nc_local + 1];

    double start_time, end_time;
    if (rank == 0) {
        start_time = MPI_Wtime(); // Mesure du temps
    }

    // Initialisation des conditions initiales pour le domaine local
    init_conditions_local(nc_local, xc_local, h_local, u_local, global_start_idx, dx, XLEFT, XM);

    // Copie des conditions initiales h_local, u_local dans W_local pour les cellules réelles locales
    for (int i = 0; i < nc_local; ++i) { // Boucle sur les indices 0 à nc_local-1 pour xc_local
        W_local[0][i + 1] = h_local[i + 1]; // Copie la hauteur
        W_local[1][i + 1] = h_local[i + 1] * u_local[i + 1]; // Copie le moment
    }

    double t = 0.0;
    while (t < TEND) {
        // 1. Échange des cellules halo pour W_local (variables conservées)

        if (rank < num_procs - 1) {
            // Dans ce cas c'est just pour le voisin de droite
            // Échange de la composante HAUTEUR (W[0])
            MPI_Sendrecv(&W_local[0][nc_local],     // sendbuf: adresse de h de la dernière cellule réelle de CE processus
                         1,                          // sendcount: on envoie 1 double (seulement h)
                         MPI_DOUBLE,                 // sendtype
                         rank + 1,                   // dest: à qui j'envoie (voisin de droite)
                         0,                          // sendtag: tag 0 pour h envoyée vers la droite
                         &W_local[0][nc_local + 1],  // recvbuf: adresse de h du halo droit de CE processus
                         1,                          // recvcount: on reçoit 1 double (seulement h)
                         MPI_DOUBLE,                 // recvtype
                         rank + 1,                   // source: de qui je reçois (voisin de droite)
                         1,                          // recvtag: tag 1 pour h reçue du voisin de droite
                         MPI_COMM_WORLD,             // comm
                         MPI_STATUS_IGNORE);         // status

            // Échange de la composante MOMENTUM (W[1])
            MPI_Sendrecv(&W_local[1][nc_local],     
                         1,                         
                         MPI_DOUBLE,                
                         rank + 1,                  
                         2,                         
                         &W_local[1][nc_local + 1], 
                         1,                         
                         MPI_DOUBLE,                
                         rank + 1,                  
                         3,                         
                         MPI_COMM_WORLD,            
                         MPI_STATUS_IGNORE);        
        }

        // Communication avec le voisin de gauche
        if (rank > 0) {
            // Échange de la composante HAUTEUR (W[0])
            MPI_Sendrecv(&W_local[0][1],           // sendbuf: adresse de h de la première cellule réelle de CE processus
                         1,                          // sendcount
                         MPI_DOUBLE,                 // sendtype
                         rank - 1,                   // dest: à qui j'envoie (voisin de gauche)
                         1,                          // sendtag: tag 1 pour h envoyée vers la gauche (doit matcher recvtag du voisin de gauche)
                         &W_local[0][0],             // recvbuf: adresse de h du halo gauche de CE processus
                         1,                          // recvcount
                         MPI_DOUBLE,                 // recvtype
                         rank - 1,                   // source: de qui je reçois (voisin de gauche)
                         0,                          // recvtag: tag 0 pour h reçue du voisin de gauche (doit matcher sendtag du voisin de gauche)
                         MPI_COMM_WORLD,             // comm
                         MPI_STATUS_IGNORE);         // status

            // Échange de la composante MOMENTUM (W[1])
            MPI_Sendrecv(&W_local[1][1],           
                         1,                        
                         MPI_DOUBLE,               
                         rank - 1,                 
                         3,                         
                         &W_local[1][0],            
                         1,                         
                         MPI_DOUBLE,                
                         rank - 1,                  
                         2,                         
                         MPI_COMM_WORLD,            
                         MPI_STATUS_IGNORE);        
        }


        // D'abord, mettre à jour TOUTES les cellules locales (réelles et halos) de h_local et u_local
        for (int i = 0; i < nc_local + 2; ++i) { 
            h_local[i] = W_local[0][i];
            u_local[i] = W_local[1][i] / (h_local[i] > 1e-6 ? h_local[i] : 1e-6);
        }

        // Ensuite, calculer max_speed_local SEULEMENT sur les cellules réelles
        double max_speed_local = 0.0;
        for (int i = 1; i <= nc_local; ++i) {
            double c = sqrt(G * h_local[i]);
            double speed = fabs(u_local[i]) + c;
            if (speed > max_speed_local)
                max_speed_local = speed;
        }

        // 3. Synchronisation globale pour le pas de temps (dt)
        double max_speed_global;
        MPI_Allreduce(&max_speed_local, &max_speed_global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        double dt = CFL * dx / max_speed_global; // Calcul du pas de temps global
        if (t + dt > TEND)
            dt = TEND - t; 
        double nu = dt / dx; 
        t += dt; 

        // 4. Calcul des flux aux interfaces des cellules locales (incluant les halos)
        rusanov_flux(nc_local, h_local, u_local, flux_local);

        // 5. Mise à jour des variables conservées (Wn_local) pour les cellules réelles locales
        for (int i = 1; i <= nc_local; ++i) { 
            for (int j = 0; j < 2; ++j) {
                // Formule de mise à jour par volumes finis
                Wn_local[j][i] = W_local[j][i] - nu * (flux_local[j][i] - flux_local[j][i - 1]);
            }
        }

        // 6. Application des Conditions aux Limites (Neumann)
        // Ces conditions sont appliquées uniquement si le processus gère la frontière globale.
        if (rank == 0) { 
            for (int j = 0; j < 2; ++j) {
                Wn_local[j][1] = Wn_local[j][2];
            }
        }
        if (rank == num_procs - 1) { 
            for (int j = 0; j < 2; ++j) {
                Wn_local[j][nc_local] = Wn_local[j][nc_local - 1];
            }
        }

        // 7. Copie des valeurs mises à jour (Wn_local) vers le tableau principal (W_local) pour la prochaine itération
        for (int i = 1; i <= nc_local; ++i) { // Boucle sur les cellules réelles locales
            W_local[0][i] = Wn_local[0][i];
            W_local[1][i] = Wn_local[1][i];
        }

        if (rank == 0) {
            printf("t = %.4f\n", t); // Affichage du temps par le processus racine
        }
    }

    // Rassemblement des résultats sur le processus racine (rank 0)
    double *h_send_buffer = (double *)malloc(nc_local * sizeof(double));
    double *u_send_buffer = (double *)malloc(nc_local * sizeof(double));

    for (int i = 0; i < nc_local; ++i) {
        h_send_buffer[i] = W_local[0][i + 1];
        u_send_buffer[i] = W_local[1][i + 1] / (W_local[0][i + 1] > 1e-6 ? W_local[0][i + 1] : 1e-6);
    }

    // Tableaux pour MPI_Gatherv (comptes de réception et décalages)
    int *recvcounts = NULL;
    int *displs = NULL;
    int *all_nc_local = NULL;

    if (rank == 0) {
        recvcounts = (int *)malloc(num_procs * sizeof(int));
        displs = (int *)malloc(num_procs * sizeof(int));
        all_nc_local = (int *)malloc(num_procs * sizeof(int));
    }

    MPI_Gather(&nc_local, 1, MPI_INT, all_nc_local, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        // Calcul des recvcounts et displs pour MPI_Gatherv
        displs[0] = 0;
        recvcounts[0] = all_nc_local[0];
        for (int i = 1; i < num_procs; ++i) {
            recvcounts[i] = all_nc_local[i];
            displs[i] = displs[i - 1] + recvcounts[i - 1];
        }
    }

    // Allocation des tableaux globaux sur le processus racine
    double *h_global = NULL;
    double *u_global = NULL;
    double *xc_global = NULL;

    if (rank == 0) {
        h_global = (double *)malloc(nc_total * sizeof(double));
        u_global = (double *)malloc(nc_total * sizeof(double));
        xc_global = (double *)malloc(nc_total * sizeof(double));
    }

    // Rassemblement des valeurs de hauteur (h)
    MPI_Gatherv(h_send_buffer, nc_local, MPI_DOUBLE,
                h_global, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rassemblement des valeurs de vitesse (u)
    MPI_Gatherv(u_send_buffer, nc_local, MPI_DOUBLE,
                u_global, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // Rassemblement des coordonnées des centres de cellules (xc)
    MPI_Gatherv(xc_local, nc_local, MPI_DOUBLE,
                xc_global, recvcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        end_time = MPI_Wtime(); // Mesure du temps de fin sur le processus racine

        // Écriture des résultats finaux dans un fichier
        FILE *f = fopen("output_mpi.dat", "w");
        if (f == NULL) {
            perror("Erreur lors de l'ouverture du fichier output_mpi.dat");
            MPI_Abort(MPI_COMM_WORLD, 1); // Arrêt de tous les processus en cas d'erreur
        }
        for (int i = 0; i < nc_total; ++i) {
            fprintf(f, "%f\t%f\t%f\n", xc_global[i], h_global[i], u_global[i]);
        }
        fclose(f);

        printf("Simulation terminée. Résultats écrits dans output_mpi.dat\n");
        printf("Temps d'exécution (MPI): %.4f secondes\n", end_time - start_time);

        free(h_global);
        free(u_global);
        free(xc_global);
        free(recvcounts);
        free(displs);
        free(all_nc_local);
    }

    free(xc_local);
    free(h_local);
    free(u_local);
    free(h_send_buffer);
    free(u_send_buffer);

    MPI_Finalize();
    return 0;
}