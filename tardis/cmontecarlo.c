#include "cmontecarlo.h"

inline int64_t line_search(double *nu, double nu_insert, int64_t number_of_lines)
{
  int64_t imin, imax;
  imin = 0;
  imax = number_of_lines - 1;
  if (nu_insert > nu[imin])
    {
      return imin;
    }
  else if (nu_insert < nu[imax])
    {
      return imax + 1;
    }
  else
    {
      return binary_search(nu, nu_insert, imin, imax) + 1;
    }
}

inline int64_t binary_search(double *x, double x_insert, int64_t imin, int64_t imax)
{
  if (x_insert > x[imin] || x_insert < x[imax])
    {
      fprintf(stderr, "Binary Search called but not inside domain. Abort!");
      exit(1);
    }
  int imid = (imin + imax) / 2;
  while (imax - imin > 2)
    {
      if (x[imid] < x_insert)
	{
	  imax = imid + 1;
	}
      else
	{
	  imin = imid;
	}
      imid = (imin + imax) / 2;
    }
  if (imax - imid == 2)
    {
      if (x_insert < x[imin + 1])
	{
	  return imin + 1;
	}
    }
  return imin;
}

inline void check_array_bounds(int64_t ioned,int64_t nrow,int64_t ncolums)
{
    if (ioned > ((ncolums + 1) * (nrow +1)))
        {
        fprintf(stderr, "Array Index Out Of Bounds");
        exit(1);
        }
}

inline void set_array_int(int64_t irow,int64_t icolums,int64_t nrow,int64_t ncolums , int64_t *array, int64_t val )
    {
    int64_t ioned = 0;
    ioned = nrow * icolums + irow;
    check_array_bounds(ioned, nrow, ncolums);
    array[ioned] = val;
    }

inline void set_array_double(int64_t  irow, int64_t icolums, int64_t nrow, int64_t ncolums , double *array, double val )
    {
    int64_t ioned = 0;
    ioned = nrow * icolums + irow;
    check_array_bounds(ioned, nrow, ncolums);
    array[ioned] = val;
    }

inline int64_t get_array_int( int64_t irow, int64_t icolums, int64_t nrow, int64_t ncolums , int64_t *array)
    {
    int64_t ioned = 0;
    ioned = nrow * icolums + irow;
    check_array_bounds(ioned, nrow, ncolums);
    return array[ioned];
    }

inline double get_array_double( int64_t irow, int64_t icolums, int64_t nrow, int64_t ncolums , double *array)
    {
    int64_t ioned = 0;
    ioned = nrow * icolums + irow;
    check_array_bounds(ioned, nrow, ncolums);
    return array[ioned];
    }

inline double rpacket_doppler_factor(rpacket_t *packet, storage_model_t *storage)
{
  return 1.0 - packet->mu * packet->r * storage->inverse_time_explosion * INVERSE_C;
}

inline double compute_distance2outer(rpacket_t *packet, storage_model_t *storage)
{
  double r = packet->r;
  double mu = packet->mu;
  double r_outer = storage->r_outer[packet->current_shell_id];
  return sqrt(r_outer * r_outer + ((mu * mu - 1.0) * r * r)) - (r * mu);
}

inline double compute_distance2inner(rpacket_t *packet, storage_model_t *storage)
{
  if (packet->recently_crossed_boundary == 1)
    {
      return MISS_DISTANCE;
    }
  double r = packet->r;
  double mu = packet->mu;
  double r_inner = storage->r_inner[packet->current_shell_id];
  double check = r_inner * r_inner + (r * r * (mu * mu - 1.0));
  if (check < 0.0) 
    {
      return MISS_DISTANCE;
    }
  else
    {
      return mu < 0.0 ? -r * mu - sqrt(check) : MISS_DISTANCE;
    }
}

inline double compute_distance2boundary(rpacket_t *packet, storage_model_t *storage)
{
  double r = packet->r;
  double mu = packet->mu;  
  double r_outer = storage->r_outer[packet->current_shell_id];
  double r_inner = storage->r_inner[packet->current_shell_id];
  double d_outer = sqrt(r_outer * r_outer + ((mu * mu - 1.0) * r * r)) - (r * mu);
  double d_inner;
  if (packet->recently_crossed_boundary == 1)
    {
      return d_outer;
    }
  else
    {
      double check = r_inner * r_inner + (r * r * (mu * mu - 1.0));
      if (check < 0.0) 
	{
	  return d_outer;
	}
      else
	{
	  d_inner =  mu < 0.0 ? -r * mu - sqrt(check) : MISS_DISTANCE;
	}
    }
  return d_inner < d_outer ? d_inner : d_outer;
}

inline double compute_distance2line(rpacket_t *packet, storage_model_t *storage)
{
  if (packet->last_line == 1)
    {
      return MISS_DISTANCE;
    }
  double r = packet->r;
  double mu = packet->mu;
  double nu = packet->nu;
  double nu_line = packet->nu_line;
  double t_exp = storage->time_explosion;
  double inverse_t_exp = storage->inverse_time_explosion;
  double last_line = storage->line_list_nu[packet->next_line_id - 1];
  double next_line = storage->line_list_nu[packet->next_line_id + 1];
  int64_t cur_zone_id = packet->current_shell_id;
  double comov_nu, doppler_factor;
  doppler_factor = 1.0 - mu * r * inverse_t_exp * INVERSE_C;
  comov_nu = nu * doppler_factor;
  if (comov_nu < nu_line)
    {
      fprintf(stderr, "ERROR: Comoving nu less than nu_line!\n");
      fprintf(stderr, "comov_nu = %f\n", comov_nu);
      fprintf(stderr, "nu_line = %f\n", nu_line);
      fprintf(stderr, "(comov_nu - nu_line) / nu_line = %f\n", (comov_nu - nu_line) / nu_line);
      fprintf(stderr, "last_line = %f\n", last_line);
      fprintf(stderr, "next_line = %f\n", next_line);
      fprintf(stderr, "r = %f\n", r);
      fprintf(stderr, "mu = %f\n", mu);
      fprintf(stderr, "nu = %f\n", nu);
      fprintf(stderr, "doppler_factor = %f\n", doppler_factor);
      fprintf(stderr, "cur_zone_id = %d\n", cur_zone_id);
      exit(1);
    }
  return ((comov_nu - nu_line) / nu) * C * t_exp;
}

inline double compute_distance2electron(rpacket_t *packet, storage_model_t *storage)
{
  if (packet->virtual_packet > 0)
    {
      return MISS_DISTANCE;
    }
  double inverse_ne = storage->inverse_electron_densities[packet->current_shell_id] * 
    storage->inverse_sigma_thomson;
  return packet->tau_event * inverse_ne;
}



inline void compute_distance2continuum(rpacket_t *packet, storage_model_t *storage)
{
    if (packet->virtual_packet > 0)
        {
        packet->d_bf = MISS_DISTANCE;
        packet->d_th = MISS_DISTANCE;
        packet->d_ff = MISS_DISTANCE;
        packet->d_cont = MISS_DISTANCE;
        packet->chi_bf = 0;
        packet->chi_th = 0;
        packet->chi_ff = 0;
        packet->chi_cont = 0;

        }

    packet->d_ff = MISS_DISTANCE;
    packet->chi_ff = 0;

    packet->chi_bf = calculate_chi_bf(packet, storage);
    packet->d_bf = packet->tau_event / packet->chi_bf;
    packet->chi_th = storage->electron_densities[packet->current_shell_id] / storage->sigma_thomson;
    packet->d_th = packet->tau_event / packet->chi_th;
    packet->chi_cont = packet->chi_th + packet->chi_ff + packet->chi_bf;
    packet->d_cont = packet->tau_event / packet->chi_cont;
}

inline double calculate_chi_bf(rpacket_t *packet, storage_model_t *storage)
{
    double bf_helper = 0;
    double nu_th;
    double l_pop_r;
    double l_pop;
    double T;
    double kB;
    double nu;
    int64_t i = 0;
    int64_t I;
    int64_t atom;
    int64_t ion;
    int64_t level;

    nu = packet->nu;
    T = storage->t_electrons[packet->current_shell_id];
    kB = storage->kB;
    I = storage->chi_bf_index_to_level_nrow; // This is equal to the number of levels

    for(i=0;i<=I;++i){
        nu_th = storage->bound_free_th_frequency[i];
        if (nu_th < nu){

            //atom = get_array_int(i,0,storage->chi_bf_index_to_level_nrow, storage->chi_bf_index_to_level_ncolum,
            //storage->chi_bf_index_to_level);
            //ion = get_array_int(i,1,storage->chi_bf_index_to_level_nrow, storage->chi_bf_index_to_level_ncolum,
            //storage->chi_bf_index_to_level);
            //level = get_array_int(i,2,storage->chi_bf_index_to_level_nrow, storage->chi_bf_index_to_level_ncolum,
            //storage->chi_bf_index_to_level);
            l_pop = get_array_double(i,packet->current_shell_id, storage->bf_level_population_nrow,
            		storage->bf_level_population_ncolum, storage->bf_level_population);
            
            l_pop_r = get_array_double(i,packet->current_shell_id, storage->bf_lpopulation_ratio_nlte_lte_nrow,
            		storage->bf_lpopulation_ratio_nlte_lte_ncolum, storage->bf_lpopulation_ratio_nlte_lte);

            //l_pop = storage->bf_level_populations[i][packet->current_shell_id];
            //l_pop_r = storage->bf_lpopulation_ratio_nlte_lte[i][packet->current_shell_id ];
            bf_helper += storage->bf_cross_sections[i] * l_pop * pow((nu_th/nu),3) * (1-l_pop_r * exp(-(H * nu)/kB /T));
        }
    }
    return bf_helper;
}


inline int64_t macro_atom(rpacket_t *packet, storage_model_t *storage)
{
  int emit, i = 0;
  double p, event_random;
  int activate_level = storage->line2macro_level_upper[packet->next_line_id - 1];
  while (emit != -1)
    {
      event_random = rk_double(&mt_state);
      i = storage->macro_block_references[activate_level] - 1;
      p = 0.0;
      do
	{
	  p += storage->transition_probabilities[packet->current_shell_id * storage->transition_probabilities_nd + (++i)];
	} while (p <= event_random);
      emit = storage->transition_type[i];
      activate_level = storage->destination_level_id[i];
    }
  return storage->transition_line_id[i];
}


inline double move_packet(rpacket_t *packet, storage_model_t *storage, 
			  double distance, int64_t virtual_packet)
{
  packet->moved = 1;
  double new_r, doppler_factor, comov_energy, comov_nu;
  doppler_factor = rpacket_doppler_factor(packet, storage);
  if (distance <= 0.0)
    {
      return doppler_factor;
    }
  new_r = sqrt(packet->r * packet->r + distance * distance + 
	       2.0 * packet->r * distance * packet->mu);
  packet->mu = (packet->mu * packet->r + distance) / new_r;
  packet->r = new_r;
  if (virtual_packet > 0)
    {
      return doppler_factor;
    }
  comov_energy = packet->energy * doppler_factor;
  comov_nu = packet->nu * doppler_factor;
  storage->js[packet->current_shell_id] += comov_energy * distance;
  storage->nubars[packet->current_shell_id] += comov_energy * distance * comov_nu;
  return doppler_factor;
}

inline void increment_j_blue_estimator(rpacket_t *packet, storage_model_t *storage,
				       double d_line, int64_t j_blue_idx)
{
  double comov_energy, r_interaction, mu_interaction, doppler_factor;
  r_interaction = sqrt(packet->r * packet->r + d_line * d_line + 
		       2.0 * packet->r * d_line * packet->mu);
  mu_interaction = (packet->mu * packet->r + d_line) / r_interaction;
  doppler_factor = 1.0 - mu_interaction * r_interaction * 
    storage->inverse_time_explosion * INVERSE_C;
  comov_energy = packet->energy * doppler_factor;
  storage->line_lists_j_blues[j_blue_idx] += comov_energy / packet->nu;
}

void init_storage_model(storage_model_t *storage, 
			double *packet_nus,
			double *packet_mus,
			double *packet_energies,
			double *r_inner,
			double *line_list_nu,
			double *output_nus,
			double *output_energies,
			double time_explosion,
			double spectrum_start_nu,
			double spectrum_end_nu,
			double spectrum_delta_nu,
			double *spectrum_virt_nu,
			double sigma_thomson,
			double *electron_densities,
			double *inverse_electron_densities,
			double *js,
			double *nubars,
			int64_t no_of_lines,
			int64_t no_of_packets)
{
  storage->packet_nus = packet_nus;
  storage->packet_mus = packet_mus;
  storage->packet_energies = packet_energies;
  storage->r_inner = r_inner;
  storage->output_nus = output_nus;
  storage->output_energies = output_energies;
  storage->time_explosion = time_explosion;
  storage->inverse_time_explosion = 1.0 / time_explosion;
  storage->spectrum_start_nu = spectrum_start_nu;
  storage->spectrum_end_nu = spectrum_end_nu;
  storage->spectrum_delta_nu = spectrum_delta_nu;
  storage->spectrum_virt_nu = spectrum_virt_nu;
  storage->sigma_thomson = sigma_thomson;
  storage->inverse_sigma_thomson = 1.0 / sigma_thomson;
  storage->electron_densities = electron_densities;
  storage->inverse_electron_densities = inverse_electron_densities;
  storage->js = js;
  storage->nubars = nubars;
  storage->no_of_lines = no_of_lines;
  storage->no_of_packets = no_of_packets;
  storage->current_packet_id = -1;
  storage->chi_bf_index_to_level;
}

int64_t montecarlo_one_packet(storage_model_t *storage, rpacket_t *packet, int64_t virtual_mode)
{
  int64_t i;
  rpacket_t virt_packet;
  double mu_bin;
  double mu_min;
  double doppler_factor_ratio;
  double weight; 
  int64_t virt_id_nu;
  if (virtual_mode == 0)
    {
      montecarlo_one_packet_loop(storage, packet, 0);
    }
  else
    {
      for (i = 0; i < packet->virtual_packet_flag; i++)
	{
	  memcpy((void *)&virt_packet, (void *)packet, sizeof(rpacket_t));
	  if (virt_packet.r > storage->r_inner[0])
	    {
	      mu_min = -1.0 * sqrt(1.0 - (storage->r_inner[0] / virt_packet.r) * (storage->r_inner[0] / virt_packet.r));
	    }
	  else
	    {
	      mu_min = 0.0;
	    }
	  mu_bin = (1.0 - mu_min) / packet->virtual_packet_flag;
	  virt_packet.mu = mu_min + (i + rk_double(&mt_state)) * mu_bin;
	  switch(virtual_mode)
	    {
	    case -2:
	      weight = 1.0 / packet->virtual_packet_flag;
	      break;
	    case -1:
	      weight = 2.0 * virt_packet.mu / packet->virtual_packet_flag;
	      break;
	    case 1:
	      weight = (1.0 - mu_min) / 2.0 / packet->virtual_packet_flag;
	      break;
	    default:
	      fprintf(stderr, "Something has gone horribly wrong!\n");
	      exit(1);	      
	    }
	  doppler_factor_ratio = 
	    rpacket_doppler_factor(packet, storage) /
	    rpacket_doppler_factor(&virt_packet, storage);
	  virt_packet.energy = packet->energy * doppler_factor_ratio;
	  virt_packet.nu = packet->nu * doppler_factor_ratio;
	  montecarlo_one_packet_loop(storage, &virt_packet, 1);
	  if ((virt_packet.nu < storage->spectrum_end_nu) && 
	      (virt_packet.nu > storage->spectrum_start_nu))
	    {
	      virt_id_nu = floor((virt_packet.nu - storage->spectrum_start_nu) / 
				 storage->spectrum_delta_nu);
	      storage->spectrum_virt_nu[virt_id_nu] += virt_packet.energy * weight;
	    }
	}
    }
}

int64_t montecarlo_propagate_outwards(rpacket_t *packet, storage_model_t *storage,
				      double distance, int64_t *reabsorbed)
{
  move_packet(packet, storage, distance, packet->virtual_packet);
  if (packet->virtual_packet > 0)
    {
      packet->tau_event += distance * storage->electron_densities[packet->current_shell_id] * 
	storage->sigma_thomson;
    }
  else
    {
      packet->tau_event = -log(rk_double(&mt_state));
    }
  if (packet->current_shell_id < storage->no_of_shells - 1)
    {
      packet->current_shell_id += 1;
      packet->recently_crossed_boundary = 1;
    }
  else
    {
      *reabsorbed = 0;
      return 1;
    }
  return 0;
}

int64_t montecarlo_propagate_inwards(rpacket_t *packet, storage_model_t *storage,
				     double distance, int64_t *reabsorbed)
{
  double comov_energy, doppler_factor, comov_nu, inverse_doppler_factor;
  move_packet(packet, storage, distance, packet->virtual_packet);
  if (packet->virtual_packet > 0)
    {
      packet->tau_event += distance * storage->electron_densities[packet->current_shell_id] * storage->sigma_thomson;
    }
  else
    {
      packet->tau_event = -log(rk_double(&mt_state));
    }
  if (packet->current_shell_id > 0)
    {
      packet->current_shell_id -= 1;
      packet->recently_crossed_boundary = -1;
    }
  else
    {
      if ((storage->reflective_inner_boundary == 0) || 
	  (rk_double(&mt_state) > storage->inner_boundary_albedo))
	{
	  *reabsorbed = 1;
	  return 1;
	}
      else
	{
	  doppler_factor = rpacket_doppler_factor(packet, storage);
	  comov_nu = packet->nu * doppler_factor;
	  comov_energy = packet->energy * doppler_factor;
	  packet->mu = rk_double(&mt_state);
	  inverse_doppler_factor = 1.0 / rpacket_doppler_factor(packet, storage);
	  packet->nu = comov_nu * inverse_doppler_factor;
	  packet->energy = comov_energy * inverse_doppler_factor;
	  packet->recently_crossed_boundary = 1;
	  if (packet->virtual_packet_flag > 0)
	    {
	      montecarlo_one_packet(storage, packet, -2);
	    }
	}
    }
  return 0;
}

int64_t move_packet_across_shell_boundary(rpacket_t *packet, storage_model_t *storage, 
					  double distance, int64_t *reabsorbed)
{
  double comov_energy, doppler_factor, comov_nu, inverse_doppler_factor;
  move_packet(packet, storage, distance, packet->virtual_packet);
  if (packet->virtual_packet > 0)
    {
      packet->tau_event += distance * 
	storage->electron_densities[packet->current_shell_id] * 
	storage->sigma_thomson;
    }
  else
    {
      packet->tau_event = -log(rk_double(&mt_state));
    }
  if (packet->next_shell_id > 1 && packet->next_shell_id < storage->no_of_shells - 2)
    {
      packet->current_shell_id = packet->next_shell_id;
      packet->recently_crossed_boundary = 
	packet->current_shell_id < packet->next_shell_id ? 1 : -1; 
    }
  else
    {
      if (packet->current_shell_id < packet->next_shell_id)
	{
	  *reabsorbed = 0;
	  return 1;
	}
      else if ((storage->reflective_inner_boundary == 0) || 
	       (rk_double(&mt_state) > storage->inner_boundary_albedo))
	{
	  *reabsorbed = 1;
	  return 1;
	}
      else
	{
	  doppler_factor = rpacket_doppler_factor(packet, storage);
	  comov_nu = packet->nu * doppler_factor;
	  comov_energy = packet->energy * doppler_factor;
	  packet->mu = rk_double(&mt_state);
	  inverse_doppler_factor = 1.0 / rpacket_doppler_factor(packet, storage);
	  packet->nu = comov_nu * inverse_doppler_factor;
	  packet->energy = comov_energy * inverse_doppler_factor;
	  packet->recently_crossed_boundary = 1;
	  if (packet->virtual_packet_flag > 0)
	    {
	      montecarlo_one_packet(storage, packet, -2);
	    }
	}
    }
  return 0;
}
int64_t montecarlo_thomson_bound_free_scatter(rpacket_t *packet, storage_model_t *storage,
				   double distance, int64_t *reabsorbed)
{
fprintf(stderr, "bla");
//*reabsorbed = 1;
return 1;
}

int64_t montecarlo_thomson_free_free_scatter(rpacket_t *packet, storage_model_t *storage,
				   double distance, int64_t *reabsorbed)
{
fprintf(stderr, "Ooups, this should not happen! Free free scattering is not implemented yet. Abort!");
exit(1);
}



int64_t montecarlo_thomson_scatter(rpacket_t *packet, storage_model_t *storage,
				   double distance, int64_t *reabsorbed)
{
  double comov_energy, doppler_factor, comov_nu, inverse_doppler_factor;
  doppler_factor = move_packet(packet, storage, distance, packet->virtual_packet);
  comov_nu = packet->nu * doppler_factor;
  comov_energy = packet->energy * doppler_factor;
  packet->mu = 2.0 * rk_double(&mt_state) - 1.0;
  inverse_doppler_factor = 1.0 / rpacket_doppler_factor(packet, storage);
  packet->nu = comov_nu * inverse_doppler_factor;
  packet->energy = comov_energy * inverse_doppler_factor;
  packet->tau_event = -log(rk_double(&mt_state));
  packet->recently_crossed_boundary = 0;
  storage->last_interaction_type[storage->current_packet_id] = 1;
  if (packet->virtual_packet_flag > 0)
    {
      montecarlo_one_packet(storage, packet, 1);
    }  
  return 0;
}

int64_t montecarlo_line_scatter(rpacket_t *packet, storage_model_t *storage,
				double distance, int64_t *reabsorbed)
{
  double comov_energy = 0.0;
  int64_t emission_line_id = 0;
  double old_doppler_factor = 0.0;
  double inverse_doppler_factor = 0.0;
  double tau_line = 0.0;
  double tau_electron = 0.0;
  double tau_combined = 0.0;
  int64_t virtual_close_line = 0;
  int64_t j_blue_idx = -1;
  if (packet->virtual_packet == 0)
    {
      j_blue_idx = packet->current_shell_id * storage->line_lists_j_blues_nd + packet->next_line_id;
      increment_j_blue_estimator(packet, storage, distance, j_blue_idx);
    }
  tau_line = storage->line_lists_tau_sobolevs[packet->current_shell_id * storage->line_lists_tau_sobolevs_nd + packet->next_line_id];
  tau_electron = storage->sigma_thomson * storage->electron_densities[packet->current_shell_id] * distance;
  tau_combined = tau_line + tau_electron;
  packet->next_line_id += 1;
  if (packet->next_line_id == storage->no_of_lines)
    {
      packet->last_line = 1;
    }
  if (packet->virtual_packet > 0)
    {
      packet->tau_event += tau_line;
    }
  else if (packet->tau_event < tau_combined)
    {
      old_doppler_factor = move_packet(packet, storage, distance, packet->virtual_packet);
      packet->mu = 2.0 * rk_double(&mt_state) - 1.0;
      inverse_doppler_factor = 1.0 / rpacket_doppler_factor(packet, storage);
      comov_energy = packet->energy * old_doppler_factor;
      packet->energy = comov_energy * inverse_doppler_factor;
      storage->last_line_interaction_in_id[storage->current_packet_id] = packet->next_line_id - 1;
      storage->last_line_interaction_shell_id[storage->current_packet_id] = packet->current_shell_id;
      storage->last_interaction_type[storage->current_packet_id] = 2;
      if (storage->line_interaction_id == 0)
	{
	  emission_line_id = packet->next_line_id - 1;
	}
      else if (storage->line_interaction_id >= 1)
	{
	  emission_line_id = macro_atom(packet, storage);
	}
      storage->last_line_interaction_out_id[storage->current_packet_id] = emission_line_id;
      packet->nu = storage->line_list_nu[emission_line_id] * inverse_doppler_factor;
      packet->nu_line = storage->line_list_nu[emission_line_id];
      packet->next_line_id = emission_line_id + 1;
      packet->tau_event = -log(rk_double(&mt_state));
      packet->recently_crossed_boundary = 0;
      if (packet->virtual_packet_flag > 0)
	{
	  virtual_close_line = 0;
	  if (packet->last_line == 0 &&
	      fabs(storage->line_list_nu[packet->next_line_id] - packet->nu_line) / 
	      packet->nu_line < 1e-7)
	    {
	      virtual_close_line = 1;
	    }
	  // QUESTIONABLE!!!
	  int old_close_line = packet->close_line;
	  packet->close_line = virtual_close_line;
	  montecarlo_one_packet(storage, packet, 1);
	  packet->close_line = old_close_line;
	  virtual_close_line = 0;
	}
    }
  else 
    {
      packet->tau_event -= tau_line;
    }
  if (packet->last_line == 0 &&
      fabs(storage->line_list_nu[packet->next_line_id] - packet->nu_line) / packet->nu_line < 1e-7)
    {
      packet->close_line = 1;
    }
  return 0;
}

inline void montecarlo_compute_distances(rpacket_t *packet, storage_model_t *storage)
{
  // Check if the last line was the same nu as the current line.
  if (packet->close_line == 1)
    {
      // If so set the distance to the line to 0.0
      packet->d_line = 0.0;
      // Reset close_line.
      packet->close_line = 0;
    }
  else
    {
      if (packet->moved)
	{
	  packet->d_inner = compute_distance2inner(packet, storage);
	  packet->d_outer = compute_distance2outer(packet, storage);
	  packet->moved = 0;
	}
      packet->d_line = compute_distance2line(packet, storage);
      packet->d_electron = compute_distance2electron(packet, storage);
      compute_distance2continuum(packet, storage);
    }
}

inline montecarlo_event_handler_t get_event_handler(rpacket_t *packet, storage_model_t *storage, double *distance)
{
  double d_inner, d_outer, d_line, d_cont;
  montecarlo_compute_distances(packet, storage);
  d_inner = packet->d_inner;
  d_outer = packet->d_outer;
  d_cont = packet->d_cont;
  d_line = packet->d_line;
  if ((d_line <= d_outer) && (d_line <= d_inner) && (d_line <= d_cont))
    {
      *distance = d_line;
      return &montecarlo_line_scatter;
    }
  else if ((d_outer <= d_inner) && (d_outer <= d_cont))
    {
      *distance = d_outer;
      return &montecarlo_propagate_outwards;
    }
  else if ((d_inner <= d_cont))
    {
      *distance = d_inner;
      return &montecarlo_propagate_inwards;
    }
  else // Look for a continues event
    {
      if ((packet->d_bf <= packet->d_ff) && (packet->d_bf <= packet->d_th))
      {
      //Do bound free
      fprintf(stderr, "Do bf");
      *distance = packet->d_bf;
      return &montecarlo_thomson_bound_free_scatter;
      }
      else if (packet->d_ff <= packet->d_th)
      {
      // do ff
      fprintf(stderr, "Do ff");
      *distance = packet->d_ff;
      return &montecarlo_thomson_free_free_scatter;
      }
      else
      {
      //do ths
      fprintf(stderr, "Do th");
      *distance = packet->d_th;
      return &montecarlo_thomson_scatter;
      }
    }
}

int64_t montecarlo_one_packet_loop(storage_model_t *storage, rpacket_t *packet, int64_t virtual_packet)
{
  int64_t reabsorbed = 0;
  packet->tau_event = 0.0;
  packet->nu_line = 0.0;
  packet->virtual_packet = virtual_packet;
  packet->moved = 1;
  // Initializing tau_event if it's a real packet.
  if (virtual_packet == 0)
    {
      packet->tau_event = -log(rk_double(&mt_state));
    }
  // For a virtual packet tau_event is the sum of all the tau's that the packet passes.
  while (1)
    {
      // Check if we are at the end of line list.
      if (packet->last_line == 0)
	{
	  packet->nu_line = storage->line_list_nu[packet->next_line_id];
	}
      montecarlo_event_handler_t event_handler;
      double distance;
      event_handler = get_event_handler(packet, storage, &distance);
      if (event_handler(packet, storage, distance, &reabsorbed))
	{
	  break;
	}
      if (virtual_packet > 0 && packet->tau_event > 10.0)
	{
	  packet->tau_event = 100.0;
	  reabsorbed = 0;
	  break;
	}
    } 
  if (virtual_packet > 0)
    {
      packet->energy = packet->energy * exp(-1.0 * packet->tau_event);
    }
  return reabsorbed;
}

void rpacket_init(rpacket_t *packet, storage_model_t *storage, double nu, double mu, double energy, int64_t virtual_packet_flag)
{
  packet->nu = nu;
  packet->mu = mu;
  packet->energy = energy;
  packet->r = storage->r_inner[0];
  packet->current_shell_id = 0;
  packet->recently_crossed_boundary = 1;
  packet->next_line_id = line_search(storage->line_list_nu, packet->nu, storage->no_of_lines);
  packet->last_line = (packet->next_line_id == storage->no_of_lines) ? 1 : 0;
  packet->close_line = 0;
  packet->virtual_packet_flag = virtual_packet_flag;
}
