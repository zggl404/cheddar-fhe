#pragma once

#include <map>
#include <string>

/**
 * @brief The code inside this file is for test purposes and are NOT secure
 * implementations.
 *
 */

#include "Random.h"
#include "core/Context.h"
#include "core/EvkMap.h"
#include "core/EvkRequest.h"

namespace cheddar {

/**
 * @brief This class provides a simple unoptimized client interface for CKKS.
 * The security of this class is not guaranteed and should not be used in
 * production. This class is intended for testing purposes only.
 *
 * @tparam word uint32_t or uint64_t
 */
template <typename word>
class UserInterface {
  using Dv = DeviceVector<word>;
  using Ct = Ciphertext<word>;
  using Pt = Plaintext<word>;
  using Evk = EvaluationKey<word>;

  static inline bool cm_populated_ = false;

 public:
  /**
   * @brief Construct a new UserInterface object. Random secrets will be sampled
   * and basic evaluation keys (multiplication / conjugation / dense-to-sparse /
   * sparse-to-dense) will be prepared automatically.
   *
   * @param context CKKS context (can be a BootContext or a Context)
   * @param defer_boot_keys Prepare dense/sparse switch keys later with PrepareBootKeys().
   * @param key_mode generate, save, or load (load never generates new secrets/keys).
   * @param key_directory Complete key bundle directory for save/load.
   */
  explicit UserInterface(ContextPtr<word> context,bool defer_boot_keys=false,
                         std::string key_mode="generate",std::string key_directory="");

  /**
   * @brief Encrypt a plaintext into a ciphertext.
   *
   * @param ctxt output ciphertext
   * @param ptxt input plaintext
   */
  void Encrypt(Ct &ctxt, const Pt &ptxt) const;

  /**
   * @brief Decrypt a ciphertext into a plaintext.
   *
   * @param ptxt output plaintext
   * @param ctxt input ciphertext
   */
  void Decrypt(Pt &ptxt, const Ct &ctxt) const;

  // Get const reference to an evaluation key
  const Evk &GetRotationKey(int rot_idx) const;
  const Evk &GetMultiplicationKey() const;
  const Evk &GetConjugationKey() const;
  const Evk &GetDenseToSparseKey() const;
  const Evk &GetSparseToDenseKey() const;

  /**
   * @brief Getter for the evaluation key map.
   *
   * @return const EvkMap<word>& const reference to the evaluation key map
   */
  const EvkMap<word> &GetEvkMap() const;

  /**
   * @brief Prepare a rotation key for the given rotation distance.
   *
   * @param rot_idx rotation distance
   * @param max_level maximum level for the rotation key (default: -1 -->
   * param_->max_level_)
   */
  void PrepareRotationKey(int rot_idx, int max_level = -1);

  /**
   * @brief Prepare rotation keys for the given EvkRequest.
   *
   * @param evk_request The request containing rotation distances and levels.
   */
  void PrepareRotationKey(const EvkRequest &evk_request);

  // Idempotent: generate/load dense/sparse Boot keys when preparation is staged.
  void PrepareBootKeys();

  // Save is explicit and occurs after the complete request set is prepared.
  // Load mode restores secrets/basic keys here, rotations/Boot keys in their
  // corresponding preparation calls; it never falls back to generation.
  void SaveKeys();
  uint64_t KeysGenerated() const {return keys_generated_;}
  uint64_t KeysLoaded() const {return keys_loaded_;}
  uint64_t KeyBytesLoaded() const {return key_bytes_loaded_;}
  uint64_t KeyBytesSaved() const {return key_bytes_saved_;}
  size_t KeyCount() const {return evk_map_.size();}
  const std::string& KeyIdentity() const {return key_identity_;}

 private:
  static inline constexpr double kErrorStandardDeviation = 3.2;
  static inline constexpr int kernel_block_dim_ = 256;

  ContextPtr<word> context_;
  Dv main_secret_;
  Dv sparse_secret_;

  EvkMap<word> evk_map_;
  bool boot_keys_prepared_=false;

  std::vector<word> all_primes_;
  std::string key_mode_,key_directory_,key_identity_;
  std::map<int,std::pair<NPInfo,int>> stored_keys_;
  bool stored_boot_keys_=false;
  uint64_t keys_generated_=0,keys_loaded_=0,key_bytes_loaded_=0,key_bytes_saved_=0;
  std::string KeyParameters() const;
  void InitializeKeyStore();
  void LoadEvaluationKey(int id,const NPInfo& required);

  DvView<word> MainSecretView(int front_ignore = 0);
  DvConstView<word> MainSecretConstView(int front_ignore = 0) const;
  DvView<word> SparseSecretView(int front_ignore = 0);
  DvConstView<word> SparseSecretConstView(int front_ignore = 0) const;

  // Initialization sequences;
  void PrepareSecrets();
  void PrepareBasicEvks();

  void PrepareEvk(int key_idx, const NPInfo &np, const Dv &encryption_secret,
                  const Dv &target_secret);

  void SampleRandomPolynomial(Dv &poly, const NPInfo &np) const;
  void SampleError(Dv &poly, const NPInfo &np) const;

  NPInfo GetNPForEvk(int max_level) const;
};

}  // namespace cheddar
