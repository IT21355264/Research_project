const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  age: { type: String, required: true },
  phoneNum: { type: String, required: true },
  password: { type: String, required: true },
});

module.exports = mongoose.model("RecognitionUsers", userSchema);