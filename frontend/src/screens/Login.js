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
  Box,
} from "@mui/material";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import LockOutlinedIcon from "@mui/icons-material/LockOutlined"; // Icon for the login form

const Login = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const res = await axios.post("http://localhost:5000/api/auth/login", { email, password });
      localStorage.setItem("token", res.data.token); // Store token in local storage

      alert("Login Successful!");
      navigate("/"); // Redirect to home page
    } catch (err) {
      alert("Invalid Credentials");
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
          Login
        </Typography>

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

        {/* Login Button */}
        <Button
          variant="contained"
          fullWidth
          onClick={handleLogin}
          sx={{
            mb: 2,
            py: 1.5,
            borderRadius: 2, // Rounded button
            fontWeight: "bold",
          }}
        >
          Login
        </Button>

        {/* Sign Up Link */}
        <Typography align="center">
          Don't have an account?{" "}
          <Link
            href="/signup"
            underline="hover"
            sx={{ fontWeight: "bold", color: "primary.main" }}
          >
            Sign Up
          </Link>
        </Typography>
      </Paper>
    </Container>
  );
};

export default Login;