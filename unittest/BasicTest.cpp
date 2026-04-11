#undef ENABLE_EXTENSION

#include <chrono>

#include "Testbed.h"

static constexpr int warm_up = 5;
using word = uint32_t;

TEST_P(Testbed32, EncodeDecode) {
  std::cout << "Encode and Decode functions exist for test purposes and their "
               "performance is not a priority."
            << std::endl;
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    GenerateRandomMessage(msg1);

    Plaintext<word> pt1;
    EncodeGPU(pt1, msg1, level);

    std::vector<Complex> res;
    Decode(res, pt1);
    CompareMessages(msg1, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, EncodeEncryptDecryptDecode) {
  std::cout << "Encode, Encrypt, Decrypt and Decode functions exist for test "
               "purposes and their performance is not a priority."
            << std::endl;
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    GenerateRandomMessage(msg1);

    Ciphertext<word> ct1;
    EncodeAndEncrypt(ct1, msg1, level);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct1);
    CompareMessages(msg1, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtAddCt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] + msg2[i]);
    }
    Ciphertext<word> ct1, ct2;
    Ciphertext<word> ct_res;

    std::string name = "CtAddCt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Add(ct_res, ct1, ct2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtAddPt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] + msg2[i]);
    }
    Ciphertext<word> ct1;
    Plaintext<word> pt2;
    Ciphertext<word> ct_res;

    std::string name = "CtAddPt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Add(ct_res, ct1, pt2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtAddConst) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    double const_value = msg2[0].real();
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] + const_value);
    }
    Ciphertext<word> ct1;
    Constant<word> const2;
    Ciphertext<word> ct_res;

    std::string name = "CtAddConst at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeConstant(const2, const_value, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Add(ct_res, ct1, const2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtSubCt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] - msg2[i]);
    }
    Ciphertext<word> ct1, ct2;
    Ciphertext<word> ct_res;

    std::string name = "CtSubCt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Sub(ct_res, ct1, ct2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtSubPt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] - msg2[i]);
    }
    Ciphertext<word> ct1;
    Plaintext<word> pt2;
    Ciphertext<word> ct_res;

    std::string name = "CtSubPt at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Sub(ct_res, ct1, pt2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtSubConst) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    double const_value = msg2[0].real();
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] - const_value);
    }
    Ciphertext<word> ct1;
    Constant<word> const2;
    Ciphertext<word> ct_res;

    std::string name = "CtSubConst at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeConstant(const2, const_value, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Sub(ct_res, ct1, const2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, NegCt) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(-msg1[i]);
    }
    Ciphertext<word> ct1;
    Ciphertext<word> ct_res;

    std::string name = "NegCt at level" + std::to_string(level);
    auto prepare_cts = [&]() { EncodeAndEncrypt(ct1, msg1, level); };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Neg(ct_res, ct1);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtMultPt) {
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * msg2[i]);
    }
    Ciphertext<word> ct1;
    Plaintext<word> pt2;
    Ciphertext<word> ct_res;

    std::string name =
        "CtMultPt (w/o rescaling) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Mult(ct_res, ct1, pt2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtMultPlainAdd) {
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2, msg3;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    GenerateRandomMessage(msg3);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * msg2[i] + msg3[i]);
    }
    Ciphertext<word> ct1, ct3, ct_res;
    Plaintext<word> pt2, pt3;

    std::string name =
        "CtMultPlainAdd (fused) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
      context_->encoder_.Encode(pt3, level,
                                DetermineScale(level) * DetermineScale(level),
                                msg3, 0);
      interface_->Encrypt(ct3, pt3);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->MultPlainAdd(ct_res, ct1, pt2, ct3);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtMultPlainAddTensorInput) {
  for (int level = 2; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2, msg3, msg4;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    GenerateRandomMessage(msg3);
    GenerateRandomMessage(msg4);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * msg2[i] * msg3[i] + msg4[i]);
    }

    Ciphertext<word> ct1, ct2, ct_acc, ct_tensor, ct_ref, ct_ref_rel, ct_res,
        ct_res_rel;
    Plaintext<word> pt3, pt4;

    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
      Encode(pt3, msg3, level);
      context_->encoder_.Encode(
          pt4, level,
          DetermineScale(level) * DetermineScale(level) * DetermineScale(level),
          msg4, 0);
      interface_->Encrypt(ct_acc, pt4);
      context_->Mult(ct_tensor, ct1, ct2);
    };

    std::string fused_name =
        "CtMultPlainAdd (tensor input, fused) at level" +
        std::to_string(level);
    __ProfileStart(fused_name, warm_up, prepare_cts());
    context_->MultPlainAdd(ct_res, ct_tensor, pt3, ct_acc);
    __ProfileEnd(fused_name);
    context_->Relinearize(ct_res_rel, ct_res, interface_->GetMultiplicationKey());

    std::string ref_name =
        "CtMultPlainAdd (tensor input, reference) at level" +
        std::to_string(level);
    __ProfileStart(ref_name, warm_up, prepare_cts());
    context_->Mult(ct_ref, ct_tensor, pt3);
    context_->Add(ct_ref, ct_ref, ct_acc);
    __ProfileEnd(ref_name);
    context_->Relinearize(ct_ref_rel, ct_ref, interface_->GetMultiplicationKey());

    std::vector<Complex> fused_res;
    DecryptAndDecode(fused_res, ct_res_rel);
    CompareMessages(true_res, fused_res, level == param_->max_level_);

    std::vector<Complex> ref_res;
    DecryptAndDecode(ref_res, ct_ref_rel);
    CompareMessages(true_res, ref_res, level == param_->max_level_);
    CompareMessages(ref_res, fused_res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtMultConst) {
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    double const_value = msg2[0].real();
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * const_value);
    }
    Ciphertext<word> ct1;
    Constant<word> const2;
    Ciphertext<word> ct_res;

    std::string name =
        "CtMultConst (w/o rescaling) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeConstant(const2, const_value, level);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->Mult(ct_res, ct1, const2);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, CtMultImaginaryUnit) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(Complex(-msg1[i].imag(), msg1[i].real()));
    }
    Ciphertext<word> ct1;
    Ciphertext<word> ct_res;

    std::string name = "CtMultImaginaryUnit at level" + std::to_string(level);
    auto prepare_cts = [&]() { EncodeAndEncrypt(ct1, msg1, level); };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->MultImaginaryUnit(ct_res, ct1);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, HMult) {
  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[i] * msg2[i]);
    }
    Ciphertext<word> ct1, ct2;
    Ciphertext<word> ct_res, ct_tmp;

    // Merged case;
    std::string name =
        "HMult(tensor + merged relin-rescale) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      EncodeAndEncrypt(ct2, msg2, level);
    };
    __ProfileStart(name, warm_up, prepare_cts(););
    context_->HMult(ct_res, ct1, ct2, interface_->GetMultiplicationKey(), true);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);

    // Non-merged case;
    name = "HMult(tensor + relinearize) at level" + std::to_string(level);
    __ProfileStart(name, warm_up, prepare_cts(););
    context_->HMult(ct_tmp, ct1, ct2, interface_->GetMultiplicationKey(),
                    false);
    __ProfileEnd(name);

    DecryptAndDecode(res, ct_tmp);
    CompareMessages(true_res, res, level == param_->max_level_);

    name = "Rescale at level" + std::to_string(level);
    __ProfileStart(name, warm_up,
                   context_->HMult(ct_tmp, ct1, ct2,
                                   interface_->GetMultiplicationKey(), false););
    context_->Rescale(ct_res, ct_tmp);
    __ProfileEnd(name);

    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, HRot) {
  int num_slots = (1 << log_degree_) / 2;
  word test_rot_dist = 1234;
  interface_->PrepareRotationKey(test_rot_dist, param_->max_level_);

  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[(i + test_rot_dist) % num_slots]);
    }
    Ciphertext<word> ct1, ct_res;
    std::string name = "HRot at level" + std::to_string(level);
    __ProfileStart(name, warm_up, EncodeAndEncrypt(ct1, msg1, level););
    context_->HRot(ct_res, ct1, interface_->GetRotationKey(test_rot_dist),
                   test_rot_dist);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, HRotMultPlainAdd) {
  int num_slots = (1 << log_degree_) / 2;
  word test_rot_dist = 1234;
  interface_->PrepareRotationKey(test_rot_dist, param_->max_level_);

  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2, msg3;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    GenerateRandomMessage(msg3);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[(i + test_rot_dist) % num_slots] * msg2[i] +
                         msg3[i]);
    }

    Ciphertext<word> ct1, ct3, ct_res;
    Plaintext<word> pt2, pt3;

    std::string name =
        "HRotMultPlainAdd (fused) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
      context_->encoder_.Encode(pt3, level,
                                DetermineScale(level) * DetermineScale(level),
                                msg3, 0);
      interface_->Encrypt(ct3, pt3);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->HRotMultPlainAdd(ct_res, ct1, pt2, ct3,
                               interface_->GetRotationKey(test_rot_dist),
                               test_rot_dist);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, HRotPMadUnsafe) {
  int num_slots = (1 << log_degree_) / 2;
  word test_rot_dist = 1234;
  interface_->PrepareRotationKey(test_rot_dist, param_->max_level_);

  for (int level = 1; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1, msg2, msg3;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    GenerateRandomMessage(msg2);
    GenerateRandomMessage(msg3);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(msg1[(i + test_rot_dist) % num_slots] * msg2[i] +
                         msg3[i]);
    }

    Ciphertext<word> ct1, ct_res;
    Plaintext<word> pt2, pt3;

    std::string name =
        "HRotPMadUnsafe (fused) at level" + std::to_string(level);
    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct1, msg1, level);
      Encode(pt2, msg2, level);
      context_->encoder_.Encode(pt3, level,
                                DetermineScale(level) * DetermineScale(level),
                                msg3, 0);
      interface_->Encrypt(ct_res, pt3);
    };

    __ProfileStart(name, warm_up, prepare_cts());
    context_->HRotPMadUnsafe(ct_res, ct1, pt2,
                             interface_->GetRotationKey(test_rot_dist),
                             test_rot_dist);
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, HRotPMadManyUnsafe) {
  int num_slots = (1 << log_degree_) / 2;
  std::vector<int> test_rot_dists = {1024, 6144, 11264};
  std::vector<int> normalized_rot;
  normalized_rot.reserve(test_rot_dists.size());
  EvkRequest req;
  for (int rot_dist : test_rot_dists) {
    req.AddRequest(rot_dist, param_->max_level_);
    int normalized = rot_dist % num_slots;
    if (normalized < 0) normalized += num_slots;
    normalized_rot.push_back(normalized);
  }

  interface_->PrepareRotationKey(req);

  std::vector<int> test_levels = {1, param_->max_level_ / 2, param_->max_level_};
  if (test_levels[1] == test_levels[0]) {
    test_levels.erase(test_levels.begin() + 1);
  }
  if (test_levels.back() == test_levels[test_levels.size() - 2]) {
    test_levels.pop_back();
  }

  for (int level : test_levels) {
    std::vector<Complex> msg_input, msg_plain, msg_acc;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg_input);
    GenerateRandomMessage(msg_plain);
    GenerateRandomMessage(msg_acc);
    true_res = msg_acc;
    for (int rot_dist : normalized_rot) {
      for (int i = 0; i < static_cast<int>(msg_input.size()); i++) {
        true_res[i] += msg_input[(i + rot_dist) % num_slots] * msg_plain[i];
      }
    }

    Ciphertext<word> ct_input, ct_loop, ct_hoisted;
    Plaintext<word> pt_acc;
    std::vector<Plaintext<word>> pts;

    auto prepare_cts = [&]() {
      EncodeAndEncrypt(ct_input, msg_input, level);
      context_->encoder_.Encode(pt_acc, level,
                                DetermineScale(level) * DetermineScale(level),
                                msg_acc, 0);
      interface_->Encrypt(ct_loop, pt_acc);
      interface_->Encrypt(ct_hoisted, pt_acc);

      pts.clear();
      pts.reserve(test_rot_dists.size());
      for (size_t i = 0; i < test_rot_dists.size(); i++) {
        pts.emplace_back();
        Encode(pts.back(), msg_plain, level);
      }
    };

    std::string loop_name =
        "HRotPMadUnsafe loop (no hoist) at level" + std::to_string(level);
    __ProfileStart(loop_name, warm_up, prepare_cts());
    for (size_t i = 0; i < test_rot_dists.size(); i++) {
      context_->HRotPMadUnsafe(ct_loop, ct_input, pts.at(i),
                               interface_->GetRotationKey(test_rot_dists.at(i)),
                               test_rot_dists.at(i));
    }
    __ProfileEnd(loop_name);

    std::vector<Complex> loop_res;
    DecryptAndDecode(loop_res, ct_loop);
    CompareMessages(true_res, loop_res, level == param_->max_level_);

    std::string hoisted_name =
        "HRotPMadManyUnsafe (hoisted fused) at level" + std::to_string(level);
    __ProfileStart(hoisted_name, warm_up, prepare_cts());
    context_->HRotPMadManyUnsafe(ct_hoisted, ct_input, pts,
                                 interface_->GetEvkMap(), test_rot_dists);
    __ProfileEnd(hoisted_name);

    std::vector<Complex> hoisted_res;
    DecryptAndDecode(hoisted_res, ct_hoisted);
    CompareMessages(true_res, hoisted_res, level == param_->max_level_);
    CompareMessages(loop_res, hoisted_res, level == param_->max_level_);
  }
}

TEST_P(Testbed32, HConj) {
  for (int level = 0; level <= param_->max_level_; level++) {
    std::vector<Complex> msg1;
    std::vector<Complex> true_res;
    GenerateRandomMessage(msg1);
    for (int i = 0; i < static_cast<int>(msg1.size()); i++) {
      true_res.push_back(std::conj(msg1[i]));
    }
    Ciphertext<word> ct1, ct_res;
    std::string name = "HConj at level" + std::to_string(level);
    __ProfileStart(name, warm_up, EncodeAndEncrypt(ct1, msg1, level););
    context_->HConj(ct_res, ct1, interface_->GetConjugationKey());
    __ProfileEnd(name);

    std::vector<Complex> res;
    DecryptAndDecode(res, ct_res);
    CompareMessages(true_res, res, level == param_->max_level_);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Cheddar, Testbed32,
    testing::Values("bootparam_30.json", "bootparam_35.json",
                    "bootparam_40.json"),
    [](const testing::TestParamInfo<Testbed32::ParamType> &info) {
      std::string param_name = info.param;
      std::replace(param_name.begin(), param_name.end(), '.', '_');
      return param_name;
    });
