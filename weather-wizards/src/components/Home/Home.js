import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import Particle from "../Particle";

function Home() {
  return (
    <section>
      <Container fluid className="home-section" id="home">
        <Particle />
        <Container className="home-content">
          
        </Container>
      </Container>
      <Home2 />
    </section>
  );
}

export default Home;
