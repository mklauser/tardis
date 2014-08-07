#ifndef TARDIS_CMONTECARLO_H
#define TARDIS_CMONTECARLO_H

#include <stdio.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "randomkit.h"

#define MISS_DISTANCE 1e99
#define C 29979245800.0
#define INVERSE_C 3.33564095198152e-11
#define H 6.6260755e-27 // erg*s, converted to CGS units from the NIST Constant Index

rk_state mt_state;

typedef enum
  {
    TARDIS_ERROR_OK,
    TARDIS_ERROR_BOUNDS_ERROR
  } TARDIS_ERROR;

/**
 * @brief A photon packet.
 */
typedef struct RPacket
{
  double nu; /**< Frequency of the packet in Hz. */
  double mu; /**< Cosine of the angle of the packet. */
  double energy; /**< Energy of the packet in erg. */
  double r; /**< Distance from center in cm. */
  double tau_event; /**< Optical depth to next event. */
  double nu_line; /**< frequency of the last line. */
  int64_t current_shell_id; /**< ID of the current shell. */
  int64_t next_line_id; /**< The index of the next line that the packet will encounter. */
  /**
   * @brief The packet has a nu red-ward of the last line.
   * It will not encounter any lines anymore.
   */
  int64_t last_line; 
  /** 
   * @brief The packet just encountered a line that is very close to the next line.
   * The next iteration will automatically make an interaction with the next line 
   * (avoiding numerical problems).
   */
  int64_t close_line;
  /** 
   * @brief The packet has recently crossed the boundary and is now sitting on the boundary. 
   * To avoid numerical errors, make sure that d_inner is not calculated. The value is -1
   * if the packed moved inwards, 1 if the packet moved outwards and 0 otherwise.
   */
  int64_t recently_crossed_boundary;
  /**
   * @brief packet is a virtual packet and will ignore any d_line or d_electron checks.
   * It now whenever a d_line is calculated only adds the tau_line to an 
   * internal float.
   */
  int64_t virtual_packet_flag;
  int64_t virtual_packet;
  double d_inner; /**< Distance to the inner shell boundary. */
  double d_outer; /**< Distance to the outer shell boundary. */
  double d_line; /**< Distance to the next possible line event. */
  double d_electron; /**< Distance to the next electron scatter event. */
  int64_t moved;
  double d_boundary; /**< Distance to shell boundary. */
  int64_t next_shell_id; /**< ID of the next shell packet visits. */
  double chi_bf;
  double chi_th;
  double chi_ff;
  double chi_cont;
  double d_bf;
  double d_th;
  double d_ff;
  double d_cont;

} rpacket_t;

typedef struct StorageModel
{
  double *packet_nus;
  double *packet_mus;
  double *packet_energies;
  double *output_nus;
  double *output_energies;
  int64_t *last_line_interaction_in_id;
  int64_t *last_line_interaction_out_id;
  int64_t *last_line_interaction_shell_id;
  int64_t *last_interaction_type;
  int64_t no_of_packets;
  int64_t no_of_shells;
  double *r_inner;
  double *r_outer;
  double *v_inner;
  double time_explosion;
  double inverse_time_explosion;
  double *electron_densities;
  double *inverse_electron_densities;
  double *line_list_nu;
  double *line_lists_tau_sobolevs;
  int64_t line_lists_tau_sobolevs_nd;
  double *line_lists_j_blues;
  int64_t line_lists_j_blues_nd;
  int64_t no_of_lines;
  int64_t line_interaction_id;
  double *transition_probabilities;
  int64_t transition_probabilities_nd;
  int64_t *line2macro_level_upper;
  int64_t *macro_block_references;
  int64_t *transition_type;
  int64_t *destination_level_id;
  int64_t *transition_line_id;
  double *js;
  double *nubars;
  double spectrum_start_nu;
  double spectrum_delta_nu;
  double spectrum_end_nu;
  double *spectrum_virt_nu;
  double sigma_thomson;
  double inverse_sigma_thomson;
  double inner_boundary_albedo;
  int64_t reflective_inner_boundary;
  int64_t current_packet_id;

  int64_t *chi_bf_index_to_level;
  int64_t chi_bf_index_to_level_nrow;
  int64_t chi_bf_index_to_level_ncolum;

  double *bf_level_population;
  int64_t bf_level_population_nrow;
  int64_t bf_level_population_ncolum;

  double *bf_lpopulation_ratio_nlte_lte;
  int64_t bf_lpopulation_ratio_nlte_lte_nrow;
  int64_t bf_lpopulation_ratio_nlte_lte_ncolum;

  double *bf_cross_sections;
  double *t_electron;
  double *bound_free_th_frequency;
  double kB;

} storage_model_t;

typedef int64_t (*montecarlo_event_handler_t)(rpacket_t *packet, storage_model_t *storage,
					      double distance, int64_t *reabsorbed);

/** Look for a place to insert a value in an inversely sorted float array.
 *
 * @param x an inversely (largest to lowest) sorted float array
 * @param x_insert a value to insert
 * @param imin lower bound
 * @param imax upper bound
 *
 * @return index of the next boundary to the left
 */
inline int64_t binary_search(double *x, double x_insert, int64_t imin, int64_t imax);

/** Insert a value in to an array of line frequencies
 *
 * @param nu array of line frequencies
 * @param nu_insert value of nu key
 * @param number_of_lines number of lines in the line list
 *
 * @return index of the next line ot the red. If the key value is redder than the reddest line returns number_of_lines.
 */
inline int64_t line_search(double *nu, double nu_insert, int64_t number_of_lines);

/** Calculate the distance to the outer boundary.
 *
 * @param r distance from the center to the packet
 * @param mu cosine of the angle the packet is moving at
 * @param r_outer distance from the center to the outer boundary
 *
 * @return distance to the outer boundary
 */
inline double compute_distance2outer(rpacket_t *packet, storage_model_t *storage);

/** Calculate the distance to the inner boundary.
 *
 * @param packet rpacket structure with packet information
 * @param storage storage model data
 *
 * @return distance to the inner boundary
 */
inline double compute_distance2inner(rpacket_t *packet, storage_model_t *storage);

/** Calculate the distance to shell boundary.
 *
 * @param packet rpacket structure with packet information
 * @param storage storage model data
 *
 * @return distance to shell boundary
 */
inline double compute_distance2boundary(rpacket_t *packet, storage_model_t *storage);

/** Calculate the distance the packet has to travel until it redshifts to the first spectral line.
 *
 * @param packet rpacket structure with packet information
 * @param storage storage model data
 *
 * @return distance to the next spectral line
 */
inline double compute_distance2line(rpacket_t *packet, storage_model_t *storage);

/** Calculate the distance to the Thomson scatter event.
 *
 * @param packet rpacket structure with packet information
 * @param storage storage model data
 *
 * @return distance to the Thomson scatter event in centimeters
 */
inline void compute_distance2continuum(rpacket_t *packet, storage_model_t *storage);

inline double calculate_chi_bf(rpacket_t *packet, storage_model_t *storage);


inline double compute_distance2electron(rpacket_t *packet, storage_model_t *storage);

inline int64_t macro_atom(rpacket_t *packet, storage_model_t *storage);

inline double move_packet(rpacket_t *packet, storage_model_t *storage, 
			  double distance, int64_t virtual_packet);

inline void increment_j_blue_estimator(rpacket_t *packet, storage_model_t *storage, 
				       double d_line, int64_t j_blue_idx);

int64_t montecarlo_one_packet(storage_model_t *storage, rpacket_t *packet, 
			      int64_t virtual_mode);

int64_t montecarlo_one_packet_loop(storage_model_t *storage, rpacket_t *packet, 
				   int64_t virtual_packet);

/**
 * @brief Initialize RPacket data structure.
 *
 * @param packet a pointer to the packet structure
 * @param storage a pointer to the corresponding storage model
 * @param nu frequency of the packet in Hz
 * @param mu cosine of the angle the packet is moving at
 * @param energy energy of the packet in erg
 * @param r distance to the packet from the center in cm
 * @param virtual_packet is the packet virtual
 */
void rpacket_init(rpacket_t *packet, storage_model_t *storage, double nu, double mu, double energy, int64_t virtual_packet);

inline void check_array_bounds(int64_t ioned, int64_t nrow, int64_t ncolums);

inline void set_array_int( int64_t irow, int64_t icolums, int64_t nrow, int64_t ncolums , int64_t *array, int64_t val );

inline void set_array_double( int64_t irow, int64_t icolums, int64_t nrow, int64_t ncolums , double *array, double val);

inline int64_t get_array_int( int64_t irow, int64_t icolums, int64_t nrow, int64_t ncolums , int64_t *array);

inline double get_array_double( int64_t irow, int64_t icolums, int64_t nrow, int64_t ncolums , double *array);

int64_t montecarlo_thomson_free_free_scatter(rpacket_t *packet, storage_model_t *storage,
				   double distance, int64_t *reabsorbed);

int64_t montecarlo_thomson_bound_free_scatter(rpacket_t *packet, storage_model_t *storage,
				   double distance, int64_t *reabsorbed);

#endif // TARDIS_CMONTECARLO_H
