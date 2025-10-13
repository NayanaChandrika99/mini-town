import Head from 'next/head'
import Map from '../components/Map'

export default function Home() {
  return (
    <>
      <Head>
        <title>Mini-Town - Day 0.5</title>
        <meta name="description" content="Mini-Town generative agents simulation" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <main>
        <Map width={800} height={600} />
      </main>
    </>
  )
}
