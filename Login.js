// src/components/Login.js
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import doctorImage from "../assets/doctor_cartoon.png"; // Local image

const Login = ({ onLogin }) => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError("");
    try {
      const response = await fetch("http://127.0.0.1:5000/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });

      const data = await response.json();

      if (response.ok) {
        localStorage.setItem("isLoggedIn", "true");
        localStorage.setItem("doctorEmail", email);
        if (onLogin) onLogin(email);
        navigate("/dashboard");
      } else {
        setError(data.error || "Login failed");
      }
    } catch (err) {
      setError("Server error. Make sure backend is running.");
    }
  };

  return (
    <div style={containerStyle}>
      <div style={cardStyle}>
        {/* LEFT PANEL */}
        <div style={leftStyle}>
          <h1 style={helloStyle}>Welcome to MedVise</h1>
          <p style={subTextStyle}>
            AI-driven Stratified Health Analytics for Precision Care
          </p>
          <img src={doctorImage} alt="Doctor Illustration" style={imageStyle} />
        </div>

        {/* RIGHT PANEL */}
        <div style={rightStyle}>
          <h2 style={logoStyle}>MedVise Login</h2>

          <form onSubmit={handleLogin} style={{ width: "100%" }}>
            <div style={fieldStyle}>
              <label style={labelStyle}>Email Address</label>
              <input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                style={inputStyle}
              />
            </div>

            <div style={fieldStyle}>
              <label style={labelStyle}>Password</label>
              <input
                type="password"
                placeholder="Enter your password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                style={inputStyle}
              />
            </div>

            {error && <p style={errorStyle}>{error}</p>}

            <button type="submit" style={buttonStyle}>
              Login
            </button>

            <p style={toggleTextStyle}>
              New doctor?{" "}
              <span style={linkStyle} onClick={() => navigate("/register")}>
                Register here
              </span>
            </p>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Login;

// ---------------- STYLES ----------------
const containerStyle = {
  display: "flex",
  justifyContent: "center",
  alignItems: "center",
  height: "100vh",
  background: `
    url('https://www.transparenttextures.com/patterns/cubes.png') repeat,
    linear-gradient(135deg, #b8dfff 0%, #6aa0d8 100%)
  `, // soft pastel gradient with subtle abstract pattern
  fontFamily: "Poppins, sans-serif",
};

const cardStyle = {
  display: "flex",
  width: "900px",
  height: "520px",
  backgroundColor: "#ffffff",
  borderRadius: "22px",
  overflow: "hidden",
  border: "2px solid #5a8dc5", // elegant border
  boxShadow: "0 18px 50px rgba(0,0,0,0.25)",
};

const leftStyle = {
  flex: 1,
  background: "linear-gradient(135deg, #dcefff, #a7c8eb)", // softer, slightly brighter
  padding: "50px",
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
  alignItems: "center",
  textAlign: "center",
};

const rightStyle = {
  flex: 1,
  padding: "50px 45px",
  display: "flex",
  flexDirection: "column",
  justifyContent: "center",
};

const fieldStyle = { marginBottom: "20px", textAlign: "left" };

const inputStyle = {
  width: "100%",
  padding: "14px",
  border: "1px solid #bbb",
  borderRadius: "12px",
  fontSize: "15px",
  outline: "none",
  boxShadow: "inset 0 2px 6px rgba(0,0,0,0.05)",
};

const labelStyle = { display: "block", marginBottom: "6px", fontSize: "14px", fontWeight: "500", color: "#555" };

const buttonStyle = {
  width: "100%",
  padding: "14px",
  backgroundColor: "#2f609b",
  color: "#fff",
  border: "none",
  borderRadius: "12px",
  fontSize: "16px",
  fontWeight: "600",
  cursor: "pointer",
  transition: "0.3s",
  boxShadow: "0 6px 20px rgba(47,96,155,0.3)",
};

const errorStyle = { color: "#D32F2F", marginBottom: "15px", fontSize: "14px", fontWeight: "500" };

const toggleTextStyle = { marginTop: "15px", fontSize: "14px", color: "#555" };

const linkStyle = { color: "#2f609b", cursor: "pointer", fontWeight: "bold" };

const helloStyle = { fontSize: "28px", fontWeight: "700", color: "#2f609b", marginBottom: "12px", textAlign: "center" };

const subTextStyle = { fontSize: "14px", color: "#555", marginBottom: "20px", textAlign: "center", lineHeight: "1.5" };

const imageStyle = { width: "400px", maxWidth: "95%", marginTop: "10px" };

const logoStyle = { marginBottom: "30px", color: "#2f609b", fontWeight: "700", fontSize: "26px" };