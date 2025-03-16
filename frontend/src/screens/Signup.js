import React, { useState } from "react";
import {
  TextField,
  Button,
  Typography,
  Paper,
  Container,
  CssBaseline,
  Avatar,
  Link,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined"; // Icon for the signup form

const Signup = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [age, setAge] = useState("");
  const [phoneNum, setPhone] = useState("");
  const navigate = useNavigate();

  const handleSignup = async () => {
    try {
      await axios.post("http://localhost:5000/api/auth/register", {
        name,
        email,
        age,
        phoneNum,
        password,
      });
      alert("Signup Successful! Please Login");
      navigate("/login");
    } catch (err) {
      alert("Error Signing Up");
    }
  };

  return (
    <Container
      component="main"
      maxWidth="xs"
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh",
      }}
    >
      <CssBaseline />
      <Paper
        elevation={4}
        sx={{
          padding: 4,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          borderRadius: 3, // Rounded corners
          backgroundColor: "#f5f5f5", // Light background color
          width: "100%",
          maxWidth: "400px",
        }}
      >
        {/* Avatar Icon */}
        <Avatar
          sx={{
            m: 2,
            bgcolor: "primary.main",
            width: 56,
            height: 56,
          }}
        >
          <LockOutlinedIcon fontSize="medium" />
        </Avatar>

        {/* Title */}
        <Typography
          component="h1"
          variant="h5"
          sx={{
            mb: 3,
            fontWeight: "bold",
            color: "primary.main",
          }}
        >
          Sign Up
        </Typography>

        {/* Name Field */}
        <TextField
          label="Name"
          fullWidth
          margin="normal"
          value={name}
          onChange={(e) => setName(e.target.value)}
          sx={{ mb: 2 }}
          InputProps={{
            sx: { borderRadius: 2 }, // Rounded input field
          }}
        />

        {/* Email Field */}
        <TextField
          label="Email"
          fullWidth
          margin="normal"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          sx={{ mb: 2 }}
          InputProps={{
            sx: { borderRadius: 2 }, // Rounded input field
          }}
        />

        {/* Age Field */}
        <TextField
          label="Age"
          fullWidth
          margin="normal"
          value={age}
          onChange={(e) => setAge(e.target.value)}
          sx={{ mb: 2 }}
          InputProps={{
            sx: { borderRadius: 2 }, // Rounded input field
          }}
        />

        {/* Contact Number Field */}
        <TextField
          label="Contact Number"
          fullWidth
          margin="normal"
          value={phoneNum}
          onChange={(e) => setPhone(e.target.value)}
          sx={{ mb: 2 }}
          InputProps={{
            sx: { borderRadius: 2 }, // Rounded input field
          }}
        />

        {/* Password Field */}
        <TextField
          label="Password"
          type="password"
          fullWidth
          margin="normal"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          sx={{ mb: 3 }}
          InputProps={{
            sx: { borderRadius: 2 }, // Rounded input field
          }}
        />

        {/* Sign Up Button */}
        <Button
          variant="contained"
          fullWidth
          onClick={handleSignup}
          sx={{
            mb: 2,
            py: 1.5,
            borderRadius: 2, // Rounded button
            fontWeight: "bold",
          }}
        >
          Sign Up
        </Button>

        {/* Login Link */}
        <Typography align="center">
          Already have an account?{" "}
          <Link
            href="/login"
            underline="hover"
            sx={{ fontWeight: "bold", color: "primary.main" }}
          >
            Login
          </Link>
        </Typography>
      </Paper>
    </Container>
  );
};

export default Signup;